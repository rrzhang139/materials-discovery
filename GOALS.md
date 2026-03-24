# Research Goals

## Current Goal: Property-Aligned Factorized Latent Spaces

### Motivation
CliqueFlowmer predicts properties additively: `f(z) = Σ g_c(Z_c)`. But nothing in training enforces that the property genuinely decomposes this way. If the true property function has strong cross-clique interactions, the additive predictor systematically misguides ES optimization.

### Approach
Train an unconstrained predictor `h(z)` on the full latent vector and penalize the factorization gap: `||h(z) - Σ g_c(Z_c)||²`. This forces the encoder to organize the latent space so additivity holds.

### Eval Criteria
- **Factorization gap**: `||h(z) - Σ g_c(Z_c)||²` on train, val, and optimized latent codes
- **Property MAE**: prediction accuracy of additive vs unconstrained predictor
- **Stitching validity**: fraction of stitched latent codes that decode to valid crystals with oracle-confirmed properties
- **SUN rate**: Stable, Unique, Novel rate under DFT validation

## Research Agenda

### Phase 1: Factorization Gap Measurement
Quantify how non-additive the property function is in the current latent space. Train h(z) post-hoc and measure the gap across training data and ES-optimized regions.

### Phase 2: Property-Aligned Training
Implement factorization loss (`alpha_fact > 0`) and train. Compare validity rate, oracle-evaluated properties, and SUN rates against baseline.

### Phase 3: Stitching Quality Audit
After ES optimization, measure what fraction of stitched latent codes decode to (a) valid crystal structures, (b) oracle-confirmed low property values, (c) DFT-validated structures.

### Phase 4: Clique Interpretability
With property-aligned training, analyze whether individual cliques specialize for composition, bonding topology, geometry, or symmetry.
