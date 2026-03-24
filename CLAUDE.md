# Materials Discovery — CliqueFlowmer

## Shared Infrastructure
- Shared research utilities: `../personal-research/`
- Data source (MP-20 CSVs): `../cdvae-data/data/mp_20/`

## Project Overview
CliqueFlowmer learns crystal structure representations via a VAE with clique-factorized latent spaces. The 121-dim latent vector decomposes into 8 overlapping cliques (16-dim each, knot_dim=1). Properties are predicted additively: `f(z) = Σ g_c(Z_c)`. New materials are designed via ES optimization in latent space, then decoded and validated with M3GNet oracle + DFT.

### Architecture Data Flow
```
Crystal → Encoder (Transformer) → z ∈ R^121 → separate_latents → Z_c ∈ R^{8×16}
                                    ↓                                    ↓
                                Decoder (Flow)                   DMLP predictor
                                    ↓                                    ↓
                            abc, angles, atoms, pos              Σ g_c(Z_c) → property
```

### Key Files
- `CliqueFlowmer/models/cliqueflowmer.py` — Encoder, Decoder, training/eval loops, predict()
- `CliqueFlowmer/models/graphops.py` — `chain_of_cliques`, `separate_latents` (clique decomposition)
- `CliqueFlowmer/architectures/backbones.py` — DMLP (additive predictor), UnconstrainedMLP, Transformer
- `CliqueFlowmer/architectures/ops.py` — MLP, SwiGLU, attention pool, etc.
- `CliqueFlowmer/optimization/learner.py` — ES optimization algorithm
- `CliqueFlowmer/optimization/design.py` — Design wrapper for latent codes
- `CliqueFlowmer/data/tools.py` — Dataset, collation, structure conversion
- `CliqueFlowmer/saving.py` — Checkpoint save/load (strict=False for backward compat)
- `CliqueFlowmer/train_local.py` — Single-GPU training
- `CliqueFlowmer/train.py` — Distributed training (DDP)
- `CliqueFlowmer/optimize.py` — Full optimization pipeline with oracle + SUN evaluation
- `CliqueFlowmer/measure_factorization_gap.py` — Diagnostic: measure factorization gap
- `CliqueFlowmer/configs/mp20/cliqueflowmer.py` — Hyperparameters

## Quick Start

### Local training
```bash
cd CliqueFlowmer
python train_local.py --batch_size=128 --N_epochs=25000 --N_eval=100 --N_save=1000
```

### Resume from checkpoint
```bash
python train_local.py --noFrom_scratch --batch_size=128
```

### Optimization (requires pretrained checkpoint + M3GNet)
```bash
python optimize.py --design_batch_size=1000 --top_k=100
```

### Measure factorization gap (diagnostic)
```bash
python measure_factorization_gap.py --batch_size=256
```

## RunPod Setup (Tested on A40, March 2026)

The `setup_pod.sh` script is outdated. Use this manual setup instead:

### 1. Clone and create venv
```bash
cd /workspace
git clone -b factorization-loss https://github.com/rrzhang139/materials-discovery.git code/materials-discovery
cd code/materials-discovery/CliqueFlowmer

# Use --system-site-packages to inherit system PyTorch (saves 10+ min)
# Only safe on SECURE pods with driver >= 550. On community cloud, use plain venv + cu121 torch.
python -m venv --system-site-packages .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
# DO NOT install torch — it's already in the system image.
# DGL must come from the cu121 wheel index.
pip install 'numpy<2' pandas 'pymatgen==2023.12.18' spglib 'ase==3.26.0' \
    ml-collections absl-py tqdm wandb matplotlib py3Dmol \
    'matgl==1.2.1' 'm3gnet==0.2.4' \
    'dgl==2.0.0' -f https://data.dgl.ai/wheels/cu121/repo.html
```

### 3. Data setup
MP-20 preprocessed data (pickle files) is included in the repo at `data/preprocessed/mp20/`. If missing:
```bash
# Get CSVs from CDVAE
cd /workspace && git clone --depth 1 https://github.com/txie-93/cdvae.git cdvae-data
cp cdvae-data/data/mp_20/*.csv /workspace/code/materials-discovery/CliqueFlowmer/data/preprocessed/mp20/
cd /workspace/code/materials-discovery/CliqueFlowmer
python preprocess_mp20.py --use_csv_targets
```

### 4. W&B and training
```bash
set -a; source /workspace/.env; set +a
wandb login "$WANDB_API_KEY"

# Launch in tmux (persists across SSH disconnects)
tmux new-session -d -s train "source .venv/bin/activate && \
  set -a && source /workspace/.env && set +a && \
  PYTHONUNBUFFERED=1 python train_local.py --batch_size=128 --N_epochs=25000 --N_eval=100 --N_save=1000 \
  2>&1 | tee /workspace/train.log"
```

### Problematic Dependencies
| Package | Issue | Fix |
|---------|-------|-----|
| `torch` | 2GB download, 10+ min in venv | Use `--system-site-packages` or pre-installed system torch |
| `dgl==2.0.0` | Not on PyPI; needs cu121 wheel index | `-f https://data.dgl.ai/wheels/cu121/repo.html` |
| `py3Dmol` | Easy to forget; imported by `data/tools.py` | Include in pip install |
| `numpy` | Must be <2 for matgl/m3gnet compat | `'numpy<2'` |
| `matgl` | Version warning on import (benign) | Ignore |

### SSH Notes
- **SCP doesn't work** through RunPod proxy (subsystem request fails). Transfer files via git push/pull instead.
- **Use `ssh -tt`** for RunPod connections (proxy requires PTY).
- **PYTHONUNBUFFERED=1** when redirecting output to log files (Python uses full buffering otherwise).
- **tmux must be installed**: `apt-get install -y tmux` (wiped on pod restart).

## Experiment Workflow
1. Push code to branch, `git pull` on pod
2. Launch training in tmux with W&B logging
3. Monitor: `ssh -tt wm-a40` then `tmux attach -t train` or `tail -f /workspace/train.log`
4. After training: update PROGRESS.md with results, git commit

## Gotchas
- **numpy<2**: Required by matgl/m3gnet. Pin with `pip install 'numpy<2'`.
- **Two-phase warmup**: KL warmup (`beta_vae`) runs first, then MSE warmup (`beta_mse`) starts after `beta_vae=1`. Both controlled by `warmup` param (default 1e5 steps).
- **Polyak target_regressor**: `target_regressor` is an EMA copy of `regressor` (tau=5e-3). Used during ES optimization, not training. Updated every training step.
- **Checkpoint path**: `models/states/CliqueFlowmer/mp20/checkpoint.pth`
- **Config object types**: ml_collections ConfigDict — must `dict()` convert before `**` unpacking.
- **Atomic embedding**: `model.encode()` expects pre-embedded atomics (`model.atomic_emb(atomic)`), unlike `model.vae()` which embeds internally.
- **strict=False loading**: `saving.load_model_state_dict` uses `strict=False` so old checkpoints load into models with new parameters (e.g., `unconstrained_regressor`).
- **alpha_fact=0 is baseline**: With `alpha_fact=0` (default), h(z) trains on `z.detach()` so the encoder is completely unaffected. Identical to original training.
