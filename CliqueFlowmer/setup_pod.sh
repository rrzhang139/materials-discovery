#!/bin/bash
# setup_pod.sh — Set up CliqueFlowmer on a RunPod GPU pod
# Run this once after SSH'ing into the pod.
set -e

echo "=== Setting up CliqueFlowmer ==="

# ---- Environment ----
cd /workspace
mkdir -p code

# Clone or update repo
if [ ! -d "code/CliqueFlowmer" ]; then
    echo "--- Cloning CliqueFlowmer ---"
    # Copy from local (via git) or clone from github
    git clone https://github.com/znowu/CliqueFlowmer.git code/CliqueFlowmer
fi

cd /workspace/code/CliqueFlowmer

# ---- Python Environment ----
echo "--- Setting up Python environment ---"
python -m venv .venv 2>/dev/null || python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Install PyTorch with CUDA (RunPod has CUDA 12.1+)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install numpy'<2' pandas pymatgen'==2023.12.18' spglib ase'==3.26.0' \
    ml-collections absl-py tqdm wandb matplotlib \
    matgl'==1.2.1' m3gnet'==0.2.4' matbench \
    py3Dmol dgl'==2.0.0' -f https://data.dgl.ai/wheels/cu121/repo.html

echo "--- Dependencies installed ---"

# ---- Data ----
echo "--- Preparing MP-20 data ---"
mkdir -p data/preprocessed/mp20

# Clone CDVAE for MP-20 data if not already present
if [ ! -f "data/preprocessed/mp20/train.csv" ]; then
    cd /workspace
    git clone --depth 1 https://github.com/txie-93/cdvae.git cdvae-data 2>/dev/null || true
    cp cdvae-data/data/mp_20/*.csv /workspace/code/CliqueFlowmer/data/preprocessed/mp20/
    cd /workspace/code/CliqueFlowmer
fi

# Preprocess (using CSV targets for now — fast)
if [ ! -f "data/preprocessed/mp20/train.pickle" ]; then
    echo "--- Preprocessing MP-20 (CSV targets) ---"
    python preprocess_mp20.py --use_csv_targets
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To train:"
echo "  source .venv/bin/activate"
echo "  python train_local.py --batch_size=128 --N_epochs=25000 --N_eval=100 --N_save=1000"
echo ""
echo "For full distributed training (multi-GPU):"
echo "  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12346 train.py --batch_size=128"
echo ""
echo "To optimize after training:"
echo "  python optimize.py --design_batch_size=100 --top_k=10"
