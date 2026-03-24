#!/bin/bash
# Full training pipeline for CliqueFlowmer on RunPod
# Usage: Set POD_ID in .env, then: nohup bash run_training.sh > /workspace/train.log 2>&1 &
#
# Configurable via env vars (set in /workspace/.env):
#   WANDB_API_KEY  — required
#   RUNPOD_API_KEY — required for auto-terminate
#   POD_ID         — required for auto-terminate
#   NGPU_OVERRIDE  — override GPU count (default: auto-detect)
#   BATCH_SIZE     — per-GPU batch size (default: 1024)
#   N_EPOCHS       — number of epochs (default: 25000)
#   BRANCH         — git branch (default: factorization-loss)
#
# Pipeline: clone → install → data → DDP test → full train → upload → terminate

set -a; source /workspace/.env; set +a

BATCH_SIZE=${BATCH_SIZE:-1024}
N_EPOCHS=${N_EPOCHS:-25000}
BRANCH=${BRANCH:-factorization-loss}

echo "[$(date)] === SETUP ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2
NGPU=${NGPU_OVERRIDE:-$(nvidia-smi -L | wc -l)}
echo "GPUs: $NGPU, batch_size: $BATCH_SIZE, epochs: $N_EPOCHS, branch: $BRANCH"

echo "[$(date)] === CLONE ==="
cd /workspace
if [ ! -d code/md ]; then
    git clone --filter=blob:none --sparse -b "$BRANCH" https://github.com/rrzhang139/materials-discovery.git code/md
    cd code/md && git sparse-checkout set CliqueFlowmer CLAUDE.md GOALS.md PROGRESS.md
else
    cd code/md && git fetch origin && git reset --hard "origin/$BRANCH"
fi
cd /workspace/code/md/CliqueFlowmer

echo "[$(date)] === VENV ==="
if [ ! -f .venv/bin/activate ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; assert torch.cuda.is_available(); print(f'torch {torch.__version__}, CUDA OK, {torch.cuda.device_count()} GPUs')"

echo "[$(date)] === DEPS ==="
pip install -q 'numpy<2' pandas 'pymatgen==2023.12.18' spglib 'ase==3.26.0' \
    ml-collections absl-py tqdm wandb matplotlib py3Dmol \
    'matgl==1.2.1' 'm3gnet==0.2.4' \
    'dgl==2.0.0' -f https://data.dgl.ai/wheels/cu121/repo.html

echo "[$(date)] === DATA ==="
mkdir -p data/preprocessed/mp20
if [ ! -f data/preprocessed/mp20/train.csv ]; then
    cd /workspace
    git clone --depth 1 https://github.com/txie-93/cdvae.git cdvae-data 2>/dev/null || true
    cp cdvae-data/data/mp_20/*.csv /workspace/code/md/CliqueFlowmer/data/preprocessed/mp20/
    cd /workspace/code/md/CliqueFlowmer
fi
[ ! -f data/preprocessed/mp20/train.pickle ] && python preprocess_mp20.py --use_csv_targets

wandb login "$WANDB_API_KEY" 2>/dev/null

echo "[$(date)] === PRE-WARM CACHES (avoids multi-GPU race conditions) ==="
mkdir -p /root/.dgl && echo '{"backend":"pytorch"}' > /root/.dgl/config.json
rm -rf /root/.cache/matgl
python -c "from matgl import load_model; load_model('M3GNet-MP-2018.6.1-Eform'); load_model('MEGNet-MP-2019.4.1-BandGap-mfi'); print('matgl+dgl caches warmed')"
export DGLBACKEND=pytorch

echo "[$(date)] === DDP TEST (1 epoch) ==="
DGLBACKEND=pytorch PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1))) torchrun \
    --nproc_per_node=$NGPU --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=12346 \
    train.py --N_epochs=1 --N_eval=3 --N_save=100 --batch_size=$BATCH_SIZE > /workspace/test.log 2>&1
TEST_EXIT=$?
echo "[$(date)] DDP test exit: $TEST_EXIT"
if [ $TEST_EXIT -ne 0 ]; then
    echo "DDP TEST FAILED"
    tail -20 /workspace/test.log
    exit 1
fi
echo "DDP TEST PASSED"

echo "[$(date)] === FULL TRAINING: $N_EPOCHS epochs, $NGPU GPU, bs=$BATCH_SIZE ==="
rm -f models/states/CliqueFlowmer/mp20/checkpoint.pth
DGLBACKEND=pytorch PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1))) torchrun \
    --nproc_per_node=$NGPU --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=12346 \
    train.py --batch_size=$BATCH_SIZE --N_epochs=$N_EPOCHS --N_eval=100 --N_save=10000
echo "[$(date)] === TRAINING DONE ==="

echo "[$(date)] === UPLOADING CHECKPOINT ==="
python -c "
import wandb, os
run = wandb.init(project='mp20', name='upload-final-checkpoint')
art = wandb.Artifact('cliqueflowmer-factloss-baseline', type='model')
ckpt_dir = 'models/states/CliqueFlowmer/mp20/'
for f in os.listdir(ckpt_dir):
    if f.endswith('.pth'):
        art.add_file(os.path.join(ckpt_dir, f))
run.log_artifact(art)
run.finish()
print('CHECKPOINT UPLOADED')
"

if [ -n "$POD_ID" ] && [ -n "$RUNPOD_API_KEY" ]; then
    echo "[$(date)] === TERMINATING POD ==="
    curl -s -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}" \
      "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
fi

echo "[$(date)] === ALL DONE ==="
