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

## RunPod Setup
```bash
bash setup_pod.sh
source .venv/bin/activate
```

## Experiment Workflow
1. Pull weights from RunPod: `scp runpod:~/code/CliqueFlowmer/models/states/CliqueFlowmer/mp20/checkpoint.pth models/states/CliqueFlowmer/mp20/`
2. Update PROGRESS.md with results
3. Git commit with experiment notes

## Gotchas
- **numpy<2**: Required by matgl/m3gnet. Pin with `pip install 'numpy<2'`.
- **Two-phase warmup**: KL warmup (`beta_vae`) runs first, then MSE warmup (`beta_mse`) starts after `beta_vae=1`. Both controlled by `warmup` param (default 1e5 steps).
- **Polyak target_regressor**: `target_regressor` is an EMA copy of `regressor` (tau=5e-3). Used during ES optimization, not training. Updated every training step.
- **Checkpoint path**: `models/states/CliqueFlowmer/mp20/checkpoint.pth`
- **Config object types**: ml_collections ConfigDict — must `dict()` convert before `**` unpacking.
- **Atomic embedding**: `model.encode()` expects pre-embedded atomics (`model.atomic_emb(atomic)`), unlike `model.vae()` which embeds internally.
- **strict=False loading**: `saving.load_model_state_dict` uses `strict=False` so old checkpoints load into models with new parameters (e.g., `unconstrained_regressor`).
