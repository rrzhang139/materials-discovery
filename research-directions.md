# Research Directions: Property-Aligned Factorized Latent Spaces for Materials Optimization

## The Core Tension in CliqueFlowmer

CliqueFlowmer's key insight is **stitching**: decomposing a 121-dim latent vector into 8 overlapping cliques and predicting properties additively as f(z) = Σ g_c(Z_c). This enables combinatorial generalization — 45,000 training materials yield ~10^37 possible clique combinations.

But the system rests on a chain of approximations:

```
Predictor (μsec) → Oracle/M3GNet (sec) → DFT (hours) → Synthesis (days)
```

ES optimizes against the cheapest, weakest link (the predictor), and the results are only as good as the predictor in the regions that matter — the **tails of the property distribution**, where training data is sparsest.

## What Compute Solves vs. What It Doesn't

**Compute solves:** better oracles, more ES steps, larger training data, more perturbations.

**Compute does NOT solve:**
- Whether stitching produces physically meaningful materials
- Whether the latent space factorization matches reality
- Whether the decoder faithfully reconstructs stitched latent codes
- The fundamental quality of the representation

## The Bottleneck: Stitching Quality

The 10^37 combinatorial stitching argument is the paper's crown jewel. But an unanswered question remains: **how many of those combinations actually decode to valid, novel, genuinely good materials?**

Currently, stitching is uncontrolled:

1. Clique 3 of material A was learned to encode *material A as a whole* — it doesn't necessarily capture a modular, reusable component
2. The knot dimensions (shared between adjacent cliques) are supposed to ensure compatibility, but nothing in the training loss explicitly enforces this
3. The decoder must reconstruct a coherent crystal from a Frankenstein latent code it has never seen

**The additive predictor f(z) = Σ g_c(Z_c) assumes the property decomposes additively. But does it?** If the true property function in latent space has strong cross-clique interactions, the additive approximation systematically misguides ES, and stitching produces candidates that score well on a lie.

## Proposed Contribution: Property-Aligned Factorized Latent Spaces

### The Problem

The VAE and the clique predictor are trained with somewhat independent objectives:

```
VAE loss:        "make z decodable back to a crystal"     ← shapes latent for RECONSTRUCTION
Predictor loss:  "make Σ g_c(Z_c) ≈ property"            ← bolted on, hopes additivity holds
```

These can fight each other. The VAE might organize the latent space such that the property is highly non-additive over cliques, and the predictor just approximates it poorly.

### The Idea

Add a training signal that explicitly forces the property to be additive over cliques in latent space. Make the encoder learn to **decompose materials into modular cliques where the property genuinely factorizes**.

### Training Objective

```
L = L_reconstruct + L_KL + L_property + λ · L_factorization
```

Train a second, unconstrained predictor h(z) alongside the additive one Σ g_c(Z_c). The factorization loss penalizes the gap between them:

```
L_factorization = || h(z) - Σ g_c(Z_c) ||²
```

This forces the encoder to organize z such that h(z) and Σ g_c(Z_c) agree — meaning the property genuinely decomposes additively. The encoder learns to place modular, separable aspects of the material into separate cliques.

### Why This Is Compute-Resistant

More GPUs don't help here. A bigger model trained with the same VAE + MSE objective won't produce a better factorization. The issue is the **training objective itself** — it doesn't incentivize the latent space to be factorizable. This is an algorithmic improvement, not a scaling improvement.

### Comparison to Alternatives

| Approach | Scales with compute? | Improves stitching? | Sample efficient? |
|---|---|---|---|
| More oracle calls during optimization | Yes (diminishing returns) | No (same latent space) | No |
| Better/larger predictor | Yes | No (still additive approximation) | Somewhat |
| **Property-aligned factorization** | **No (algorithmic gain)** | **Yes (by construction)** | **Yes (smoother landscape)** |
| Learning clique topology | No | Partially | Somewhat |
| Multi-objective extension | Orthogonal | No | No |

## Research Agenda

### Empirical Contributions

1. **Factorization gap measurement**: Quantify how non-additive the property function is in the current latent space. Train an unconstrained predictor h(z) and measure ||h(z) - Σ g_c(Z_c)|| across the training set and in the optimization regions ES explores. This tells us how much room there is for improvement.

2. **Stitching quality audit**: After ES optimization, measure what fraction of stitched latent codes decode to (a) valid crystal structures, (b) structures with oracle-confirmed low property values, (c) structures that pass DFT validation. This establishes the baseline quality of uncontrolled stitching.

3. **Property-aligned training**: Implement the factorization loss and compare:
   - Validity rate of decoded stitched materials
   - Oracle-evaluated property values (formation energy, band gap)
   - S.U.N. rates under DFT validation
   - Number of ES steps needed to reach equivalent property values (sample efficiency)

4. **Clique interpretability**: With property-aligned training, do individual cliques capture interpretable physical aspects? Analyze whether specific cliques specialize for composition, bonding topology, geometry, or symmetry.

### Theoretical Contributions

1. **Approximation bounds**: Under what conditions on the encoder and data distribution is the additive approximation error bounded? When does stitching provably generalize vs. hallucinate?

2. **Connection to ICA**: Property-aligned factorization is analogous to Independent Component Analysis — finding a decomposition where the target function is separable. Formalize this connection and leverage ICA theory for guarantees.

3. **Compositional generalization theory**: Establish when clique-based MBO achieves genuine compositional generalization. This would be the first formal treatment of when offline optimization with factorized surrogates works.

### The Broader Principle

Representation learning for optimization is fundamentally different from representation learning for reconstruction. The field has been using VAE latent spaces (optimized for reconstruction) and hoping they're good for optimization. That's not guaranteed.

The right latent space for optimization is one where the objective function has the structure you want to exploit. This principle applies beyond materials to any MBO domain with compositional structure — proteins, molecules, metamaterials, drug design.

## Key Files in CliqueFlowmer Codebase

- `models/cliqueflowmer.py` — Encoder, decoder, predictor, training loop
- `models/graphops.py` — Clique decomposition (`chain_of_cliques`, `separate_latents`)
- `architectures/backbones.py` — DMLP (additive clique predictor)
- `optimization/learner.py` — ES algorithm
- `optimize.py` — Full optimization pipeline with oracle evaluation
- `data/tools.py` — Dataset, collation, structure unpacking
