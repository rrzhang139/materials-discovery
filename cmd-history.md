# The Quest to Design Materials with Machines: A History of AI-Driven Computational Materials Discovery

*From graph neural networks to CliqueFlowmer — how the field learned to stop sampling and start optimizing.*

---

## Prologue: The Curse of Infinite Possibility

Imagine you're standing in a library with 10^60 books. Each book describes a crystalline material — a specific arrangement of atoms in a repeating lattice that tiles infinitely through three-dimensional space. Some of these books contain the blueprint for a room-temperature superconductor. Others describe a catalyst that could make green hydrogen dirt cheap. A few hold the key to batteries that charge in seconds and last decades.

The problem is that almost every book in this library is gibberish — thermodynamically unstable arrangements that would decompose the instant you tried to synthesize them. The books with real, useful materials are scattered randomly on shelves that stretch to infinity. And you have no index.

This is computational materials discovery (CMD). The search space is not just large — it is *combinatorially vast* and *structurally hostile*. A crystalline material is defined by:

- **Lattice geometry**: three axis lengths $(a, b, c)$ and three inter-axis angles $(\alpha, \beta, \gamma)$ defining a parallelepiped
- **Composition**: which elements from the periodic table occupy the unit cell
- **Atom positions**: where each atom sits, in fractional coordinates $[0, 1)^3$
- **Number of atoms**: which varies from material to material

This gives a *transdimensional*, *mixed discrete-continuous* search space. Different materials live in different-dimensional spaces. You can't straightforwardly define a distance metric between a 4-atom binary compound and a 17-atom ternary oxide. You can't take a gradient "toward better materials" because the space isn't even fixed-dimensional.

And lurking beneath all of this is the brutal constraint of thermodynamic stability: a material must not only have desirable properties, it must be the *lowest-energy arrangement* of its constituent atoms among all competing phases. A material with a spectacular band gap is worthless if a different arrangement of the same elements is 0.2 eV/atom more stable — nature will spontaneously rearrange your creation into the boring alternative.

This is the problem that an entire generation of AI researchers set out to solve.

---

## Chapter 1: Learning to See Materials — Graph Neural Networks (2017–2022)

### The Representation Problem

Before you can generate or optimize materials, you need to *predict their properties*. For a century, the gold standard was density functional theory (DFT) — a quantum-mechanical calculation that, given atomic positions, numerically solves the Schrödinger equation to compute energies, band structures, forces, and more. DFT is accurate but agonizingly slow: a single calculation on a modest unit cell can take hours to days on a supercomputer.

The first question AI asked about materials was deceptively simple: **can we learn to predict material properties from structure, without running DFT?**

The answer required solving a representation problem. Images have pixels on a grid. Text has tokens in a sequence. But crystals have atoms at *arbitrary* positions in continuous 3D space, repeating infinitely under periodic boundary conditions. There's no natural grid, no natural sequence. And the representation must respect physical symmetries: rotating a crystal doesn't change its properties (SO(3) invariance), permuting identical atoms doesn't change anything (permutation invariance), and translating the unit cell origin is arbitrary (translation invariance).

### SchNet (2017): Continuous Filters for Atoms

**Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller. NeurIPS 2017.**

The breakthrough insight was to borrow from graph neural networks but replace discrete message-passing with *continuous-filter convolutions*. In SchNet, atoms are nodes and their pairwise distances define edges. But instead of discrete edge types, SchNet passes messages through radial basis functions of the interatomic distance — a continuous, rotationally invariant representation.

This was the conceptual leap: treat a crystal as a point cloud with learned interactions, not as a fixed graph with hand-crafted features. SchNet achieved state-of-the-art on molecular property prediction (QM9 benchmark) and opened the door for everything that followed.

**The core formulation**: Given atom types $\{Z_i\}$ and positions $\{r_i\}$, learn interaction functions $W(||r_i - r_j||)$ that aggregate neighborhood information through continuous convolutions. The output is invariant to rotation and translation by construction.

### CGCNN (2018): The Foundation for Crystalline Materials

**Xie & Grossman. Physical Review Letters, 2018.**

While SchNet targeted molecules, Tian Xie and Jeffrey Grossman asked the crystalline-specific question: can we build a GNN that works directly on the periodic crystal graph? CGCNN (Crystal Graph Convolutional Neural Network) was the answer — and it became the foundation for an entire subfield.

The key innovation was constructing the crystal graph with periodic boundary conditions: atoms in the unit cell are nodes, and edges connect atoms within a cutoff radius, *including images in neighboring unit cells*. This naturally encodes periodicity without explicitly modeling the infinite lattice.

CGCNN predicted 8 DFT-calculated properties across 46,000 materials with accuracy rivaling much more expensive descriptor-based methods. More importantly, it was *interpretable*: the learned atom features captured chemical environment information that correlated with known physical intuitions.

**Why this mattered for CMD**: CGCNN proved that you could train a single neural network to replace thousands of DFT calculations. This created the possibility of *cheap property evaluation* — a prerequisite for any optimization-based approach to materials discovery.

### MEGNet (2019): Adding Global Context

**Chen, Ye, Zuo, Zheng, Ong. Chemistry of Materials, 2019.**

CGCNN treated materials as local — each atom sees its neighbors, and global properties emerge from pooling. MEGNet (MatErials Graph Network) recognized that some properties depend on *global state* — temperature, pressure, entropy — that can't be inferred from local structure alone.

MEGNet introduced a three-level message-passing scheme: atom attributes, bond attributes, and *global state attributes* all update each other iteratively. The global state vector gives the network access to thermodynamic conditions, enabling a unified framework for both molecules and crystals.

The learned element embeddings were particularly revealing: they organized themselves to capture periodic table trends (electronegativity, atomic radius, ionization energy) without being told about chemistry. The network had *rediscovered the periodic table* from data alone.

**Connection to CliqueFlowmer**: MEGNet's band gap predictor serves as one of the two property oracles that CliqueFlowmer optimizes against. When CliqueFlowmer says "minimize band gap," it means "find materials that MEGNet predicts to have near-zero band gap."

### M3GNet (2022): A Universal Potential

**Chen & Ong. Nature Computational Science, 2022.**

The culmination of the GNN-for-materials thread was M3GNet — a *universal interatomic potential* trained on the Materials Project's massive database of structural relaxation trajectories. M3GNet covers 89 elements of the periodic table and can predict forces, energies, and stresses for essentially any inorganic material.

The scale was unprecedented: trained on ~187,000 structural relaxations comprising millions of atomic configurations, M3GNet learned a single model that could relax structures (find their lowest-energy geometry), run molecular dynamics, and predict formation energies — all at a fraction of DFT's cost.

M3GNet screened 31 million hypothetical structures and identified 1.8 million potentially stable materials. Of the top 2,000, 1,578 were confirmed by DFT. This was the first demonstration that ML surrogates could genuinely *discover* materials at scale.

**Connection to CliqueFlowmer**: M3GNet provides the formation energy oracle and the structural relaxation engine. Every material that CliqueFlowmer generates is relaxed by M3GNet, and its formation energy is evaluated by M3GNet, before the best candidates are sent to DFT for ground-truth validation.

### ALIGNN (2021): Angular Information Matters

**Choudhary & DeCost. npj Computational Materials, 2021.**

A persistent limitation of CGCNN and MEGNet was that they only captured pairwise (two-body) interactions. But many material properties depend critically on *angular* (three-body) information — bond angles determine crystal symmetry, orbital overlap, and mechanical behavior.

ALIGNN (Atomistic Line Graph Neural Network) solved this elegantly: alongside the bond graph, it constructs the *line graph* (where edges of the original graph become nodes, connected if they share a vertex). Message passing on the line graph captures triplet interactions — bond angles — without explicitly computing expensive three-body features.

The result was state-of-the-art on 52 properties across multiple databases. The key lesson: **the right inductive bias matters**. Simply making GNNs bigger doesn't help as much as giving them the right structural information.

### The Thread So Far

By 2022, the materials science community had a powerful toolkit of property predictors:

```
SchNet (2017) → CGCNN (2018) → MEGNet (2019) → ALIGNN (2021) → M3GNet (2022)
                                     ↓                              ↓
                              band gap oracle                formation energy oracle
                              (for CliqueFlowmer)            (for CliqueFlowmer)
```

These models could evaluate material properties hundreds of thousands of times faster than DFT. But they were *passive* — they could score materials, not design them. The field needed a way to *generate* new materials with desired properties. That required a fundamentally different kind of model.

---

## Chapter 2: Learning to Dream Materials — Generative Models (2020–2025)

### The Generation Problem

Predicting properties of *known* materials is useful but limited. The real prize is *inverse design*: given a target property value (say, band gap = 1.4 eV for optimal solar absorption), generate a material that achieves it.

This is an inverse problem, and it's ill-posed in the most classical sense: many materials can share the same property value, but no simple function maps from property to structure. The forward map (structure → property) is many-to-one; the inverse (property → structure) is one-to-many.

The deep learning community's instinct was to reach for generative models — learn the distribution of existing materials, then sample from it. This instinct produced a remarkable sequence of increasingly sophisticated architectures, each solving a different piece of the crystal generation puzzle.

### Early Approaches: GANs and Inverse Design (2020)

**Chen & Gu. Advanced Science, 2020.**

The earliest deep learning approaches to inverse materials design used straightforward strategies: train a forward model (structure → property), then *backpropagate through it* to find structures with desired properties. Some used GANs to generate candidate structures with active learning loops for refinement.

These methods worked for simple, low-dimensional design spaces (metamaterials, 2D lattice structures) but couldn't handle the full complexity of 3D crystals. The fundamental issue: backpropagation through a learned surrogate is fragile. The gradient of the surrogate is only meaningful near the training data — far from it, you're optimizing an artifact of the model's extrapolation, not a real physical quantity.

This failure mode — the *adversarial input problem* — would haunt the field for years and become a central motivation for CliqueFlowmer's use of evolution strategies over backpropagation.

### CDVAE (2022): The First Complete Crystal Generator

**Xie, Fu, Ganea, Barzilay, Jaakkola. ICLR 2022.**

CDVAE (Crystal Diffusion Variational Autoencoder) was the watershed moment. For the first time, a single model could generate complete, novel crystal structures — atom types, positions, and lattice parameters — from scratch.

The architecture was a two-stage pipeline:
1. A **VAE** encodes crystal structures into a latent space and decodes the composition (which elements, how many atoms) and lattice parameters
2. A **score-matching diffusion process** iteratively refines atom positions from noise, conditioned on the composition and lattice

The diffusion backbone used GemNet, an SE(3)-equivariant GNN that respects the rotational and translational symmetries of 3D space. This was essential: without equivariance, the model would waste capacity learning that rotated versions of the same crystal are identical.

**CDVAE's enduring contribution was the MP-20 benchmark**: 45,000 materials from the Materials Project with up to 20 atoms per unit cell. This dataset, and the evaluation metrics CDVAE introduced (validity, coverage, property statistics), became the standard that every subsequent model — including CliqueFlowmer — measures against.

**The limitation**: CDVAE was trained with maximum likelihood (well, the VAE/diffusion equivalents). It learned to reproduce the *distribution* of known materials. It could generate materials that *looked like* training data, but it had no mechanism to preferentially generate materials with extreme property values. If you wanted materials with very low band gap, you had to generate many candidates and filter — a strategy that scales poorly when the target region is rare in the training distribution.

### DiffCSP (2023): Diffusion Done Right for Crystals

**Jiao, Huang, Lin, Han, Chen, Lu, Liu. NeurIPS 2023.**

DiffCSP made a critical observation: CDVAE worked in *Cartesian* coordinates, but crystals naturally live in *fractional* coordinates $[0, 1)^3$ relative to the lattice vectors. This matters because periodicity is trivial in fractional coordinates (just take values modulo 1) but awkward in Cartesian coordinates (you need to explicitly handle periodic images).

DiffCSP jointly diffused lattice parameters and fractional coordinates using a periodic-E(3)-equivariant denoising network. The fractional coordinate representation meant the model naturally respected periodic boundary conditions without special handling.

The result was a significant improvement in crystal structure prediction — given a composition, predict the correct structure — and competitive unconditional generation.

### DiffCSP++ (2024): The Symmetry Imperative

**Jiao, Huang, Liu, Zhao, Liu. ICLR 2024.**

DiffCSP++ added a crucial piece of physics: *space group symmetry*. Of the 230 possible space groups, most real materials fall into a handful of common ones. DiffCSP++ constrained the diffusion process to respect space group symmetries by:

1. Expressing lattice matrices in terms of space-group-compatible basis matrices
2. Constraining atom positions to Wyckoff positions (the symmetry-allowed sites within each space group)

This dramatically improved structural accuracy — match rates jumped from 52% to 98% on simple perovskites. The lesson: **the more physics you encode in the architecture, the less the model has to learn from data**.

### FlowMM (2024): Riemannian Geometry for Crystal Generation

**Miller, Chen, Sriram, Wood. ICML 2024.**

FlowMM brought the flow matching framework (Lipman et al., 2022) to crystal generation, replacing diffusion with continuous normalizing flows on Riemannian manifolds.

The key insight was geometric. Atom positions in a crystal live on a *flat torus* $\mathbb{T}^3 = [0,1)^3$ (because of periodicity). Lattice matrices live in the space of positive-definite matrices. These are not flat Euclidean spaces — they have curvature, boundaries, and topology that diffusion models on $\mathbb{R}^n$ don't naturally respect.

FlowMM defined flow matching on these proper manifolds, learning velocity fields that transport a simple prior distribution to the data distribution along geodesics. The result was ~3x more efficient than diffusion approaches at finding stable materials.

**Connection to CliqueFlowmer**: FlowMM provides the geometric methodology for CliqueFlowmer's geometry decoder. Benjamin Kurt Miller, FlowMM's first author, is a co-author of CliqueFlowmer. The key difference: in FlowMM, the flow *is* the generative model; in CliqueFlowmer, the flow is a *decoder* within an MBO framework, conditioned on an optimized latent code rather than generating unconditionally.

### MatterGen (2025): Industrial Scale

**Zeni, Pinsler, Zügner, et al. Nature, 2025.**

MatterGen, from Microsoft Research, brought crystal generation to industrial scale. Trained on 608,000 stable materials from both the Materials Project and the Alexandria database, MatterGen used a diffusion model that jointly generated atom types, positions, and lattice parameters.

The key innovation was *property-conditional generation*: by fine-tuning with property labels, MatterGen could steer generation toward materials with desired chemistry, symmetry, or property ranges. Generated structures were 2x more likely to be novel and stable, and 10x closer to local energy minima than prior methods.

MatterGen demonstrated real-world impact: it designed TaCr₂O₆ targeting a bulk modulus of 200 GPa, which was then experimentally synthesized and confirmed. This was the first time an AI-designed crystal from a general-purpose generative model was validated in the lab.

**But here's the catch**: MatterGen's property conditioning steers the *distribution* of generated materials toward favorable regions, but it doesn't *optimize*. It's still sampling from a (conditional) distribution, not finding the material that maximizes a target property. The distinction is subtle but fundamental — and it's exactly the gap that CliqueFlowmer exploits.

### CrystalFormer (2024–2025): Transformers Enter the Arena

**Taniai et al., ICLR 2024 (encoder); Cao, Luo, Lv, Wang. Science Bulletin, 2025 (generator).**

CrystalFormer brought the transformer revolution to crystals. The encoder variant introduced "infinitely connected attention" — a way to handle the fact that in a crystal, every atom interacts with infinitely many periodic images of every other atom. This is formalized as a *neural potential summation*, making the infinite attention series tractable.

The generative variant autoregressively produces atoms one at a time, conditioned on space group information. This autoregressive approach naturally handles variable numbers of atoms — a key advantage over fixed-architecture diffusion models.

**Connection to CliqueFlowmer**: CliqueFlowmer's atom-type decoder is autoregressive (like CrystalFormer), but its geometry decoder uses flows (like FlowMM). The paper explicitly notes: "We generate atom species autoregressively in our model, but the geometry is generated with flow models. Most importantly, in our MBO model, these components are not the core generators, but they are just decoders of latent representations of optimized materials."

### The Generative Paradigm's Fundamental Limitation

By 2025, the generative models for crystals had become remarkably sophisticated. But they all shared a deep structural limitation that no amount of architectural innovation could fix:

**They were density estimators, not optimizers.**

Maximum likelihood training (and its diffusion/flow equivalents) learns to place probability mass where data is dense. The model concentrates its generative capacity on reproducing the *typical* materials in the training set. Materials with extreme property values — the ones you actually want for applications — are rare in the training distribution and therefore rarely generated.

You can condition on properties (as MatterGen does), but conditioning only shifts the distribution; it doesn't enable aggressive extrapolation beyond the training range. The model has no incentive to explore the tails of the property distribution because those tails contribute negligibly to the training loss.

This is the fundamental tension: **generative models explore the known materials space, but CMD needs methods that explore *attractive regions* beyond it.**

The resolution would come from a completely different intellectual tradition.

---

## Chapter 3: Learning to Optimize Offline — Model-Based Optimization (2020–2024)

### A Parallel Thread from Reinforcement Learning

While the materials science community was building better generative models, a separate community — rooted in reinforcement learning and Bayesian optimization — was developing tools for a different but structurally identical problem: **offline model-based optimization (MBO)**.

The setup: you have a dataset of designs $\{(x_i, y_i)\}$ where $x_i$ is a design (a protein sequence, a robot morphology, a superconductor composition) and $y_i$ is its measured performance. You want to find a new design $x^*$ that maximizes $y$ — but you cannot run new experiments. You must work entirely offline, using only the existing dataset.

This is exactly the CMD problem in disguise. The "designs" are materials, the "performance" is a target property, and the "offline" constraint reflects the reality that DFT calculations are expensive and synthesis even more so. You want to leverage existing databases (like the Materials Project) to find optimal materials without running millions of new calculations.

### Model Inversion Networks (2020): The Naive Approach and Its Failure

**Kumar & Levine. NeurIPS 2020.**

The simplest MBO approach: train a forward model $f_\theta(x) \approx y$, then invert it — find $x^*$ that maximizes $f_\theta(x)$ via gradient ascent.

MINs (Model Inversion Networks) made this slightly more sophisticated by training a conditional generative model $p(x | y)$ that maps from property values to designs. Given a target $y^*$, you sample $x^* \sim p(x | y^*)$.

The problem with both approaches is **distributional shift**. The forward model $f_\theta$ is only accurate near the training data. Gradient ascent on $f_\theta$ doesn't find designs with genuinely high $y$ — it finds designs that exploit errors in $f_\theta$'s extrapolation. These *adversarial inputs* can have arbitrarily high predicted performance while being physically meaningless.

This is the same pathology that plagues offline reinforcement learning: a learned Q-function overestimates the value of out-of-distribution actions, causing the policy to choose those actions and fail catastrophically. The parallel is not coincidental — the same research group (Sergey Levine's lab at UC Berkeley) was working on both problems simultaneously.

### Conservative Objective Models (2021): Learning to Be Pessimistic

**Trabucco, Kumar, Geng, Levine. ICML 2021.**

COMs addressed the adversarial input problem directly. The key idea: train a *conservative* surrogate that deliberately underestimates performance for out-of-distribution inputs. This is achieved by adding two regularization terms to the training loss:

1. **Maximize predictions on training data** (the model should be accurate where it has evidence)
2. **Minimize predictions on adversarially generated designs** (the model should be pessimistic where it's uncertain)

The optimizer then maximizes this conservative surrogate. Because the surrogate is pessimistic out-of-distribution, the optimizer is incentivized to stay near the data while still finding the best designs within the high-confidence region.

This is philosophically identical to Conservative Q-Learning (CQL) in offline RL — the insight that pessimism under uncertainty is a better default than optimism when you can't collect new data. It's a profound inversion of the usual Bayesian optimization philosophy (which uses *upper* confidence bounds to encourage exploration). In offline settings, you can't explore, so optimism is dangerous.

### Design-Bench (2022): Standardizing the Problem

**Trabucco, Geng, Kumar, Levine. ICML 2022.**

Design-Bench provided what every nascent field needs: a standardized benchmark. It packaged diverse offline optimization tasks — protein fluorescence, DNA enhancer activity, superconductor critical temperature, robot morphology — into a unified evaluation framework.

Design-Bench revealed several uncomfortable truths:
- Many MBO methods barely outperformed simple baselines
- Performance varied wildly across tasks — no method dominated
- The superconductor task (the closest to CMD) was among the hardest

But Design-Bench also had a limitation that would prove crucial: all its tasks had **fixed-dimensional** inputs. Proteins were represented as fixed-length sequences, superconductors as fixed-length composition vectors. None of the tasks involved the *transdimensional*, mixed discrete-continuous structure of crystalline materials.

This meant that all MBO methods developed on Design-Bench — including COMs, MINs, and their successors — assumed a fixed-dimensional continuous input space. They literally could not be applied to CMD without fundamental architectural changes.

### Functional Graphical Models (2024): The Theoretical Foundation

**Grudzien, Uehara, Levine, Abbeel. AISTATS 2024.**

This paper laid the theoretical groundwork for clique-based optimization. The key insight was that if the objective function decomposes as a sum over local factors (a "functional graphical model"), then optimization can exploit this structure through dynamic programming and compositional search.

The connection to graphical models in statistics is deep. In a Markov random field, the joint distribution factorizes over cliques of the graph, and inference (computing marginals, finding the MAP configuration) is tractable when the graph has low treewidth. Similarly, if the objective function factorizes over cliques of a latent representation, optimization is tractable — you can find the optimal configuration of each clique independently (subject to overlap constraints) and compose them.

This is the mathematical formalization of **stitching**: combining locally optimal sub-solutions into globally competitive solutions. The term comes from offline RL, where "trajectory stitching" means assembling a good trajectory from good segments of different training trajectories. Here, it's "design stitching" — assembling a good design from good components of different training designs.

### Cliqueformer (2024): Stitching Meets Transformers

**Kuba, Abbeel, Levine. arXiv:2410.13106.**

Cliqueformer operationalized the functional graphical model theory. The architecture:

1. Encode designs into a latent vector $z$
2. Reshape $z$ into a chain of overlapping cliques: $z = [Z_1, Z_2, \ldots, Z_K]$ where adjacent cliques share "knot" dimensions
3. Predict the objective as $f(z) = \sum_{c=1}^{K} g_c(Z_c)$ — an additive decomposition over cliques
4. Optimize $z$ to maximize $f(z)$, exploiting the additive structure

The additive decomposition is the crucial architectural choice. It means that the contribution of clique $c$ to the total objective doesn't depend on the other cliques (except through the shared knots). This enables combinatorial search: if you have $N$ training examples, each contributing a "version" of each clique, you have $N^K$ possible combinations — an exponential expansion of the effective search space from $N$ data points.

Cliqueformer achieved state-of-the-art on Design-Bench, particularly on the superconductor task — the closest existing benchmark to CMD.

But it still operated in fixed-dimensional input spaces. To reach CMD, it needed a partner that could handle the transdimensional, structured, physics-laden world of crystalline materials.

---

## Chapter 4: The Convergence — CliqueFlowmer (2026)

### Three Threads Become One

**Kuba, Miller, Levine, Abbeel. arXiv:2603.06082, March 2026.**

CliqueFlowmer is the convergence of the three threads:

```
Thread 1: GNN Property Predictors (2017–2022)
  SchNet → CGCNN → MEGNet → ALIGNN → M3GNet
  → Cheap, accurate property oracles

Thread 2: Generative Models for Crystals (2020–2025)
  Chen & Gu → CDVAE → DiffCSP → DiffCSP++ → FlowMM → MatterGen → CrystalFormer
  → Architectures that can encode/decode crystal structures

Thread 3: Offline Model-Based Optimization (2020–2024)
  MINs → COMs → Design-Bench → Functional Graphical Models → Cliqueformer
  → Optimization frameworks that work with offline data
```

The insight is deceptively simple: **use Thread 2's encoder/decoder technology to map crystals into a fixed-dimensional latent space, then use Thread 3's optimization framework to find optimal latent codes, then use Thread 1's oracles to evaluate the results.**

But making this work required solving a cascade of technical challenges, each informed by hard-won lessons from the prior literature.

### The Architecture, Seen Through History

**The Encoder** draws from CDVAE's idea of encoding crystals into a latent space, but replaces the GNN backbone with a transformer (inspired by CrystalFormer) and adds attention-based pooling to handle variable atom counts. The VAE-style stochastic latent ($z \sim \mathcal{N}(\mu, \sigma^2)$) ensures the latent space has density everywhere, preventing the optimizer from falling into decoder-blind regions.

**The Predictor** is pure Cliqueformer: reshape $z$ into 8 overlapping cliques of dimension 16, predict the target property as their sum. This is where stitching happens — where the 45,000 training materials spawn $10^{37}$ combinatorial possibilities.

**The Atom-Type Decoder** uses autoregressive generation (like CrystalFormer), with start/stop tokens handling variable atom counts and AdaLN conditioning (from DiT, the diffusion transformer architecture) injecting latent information.

**The Geometry Decoder** uses flow matching (from FlowMM), but with a critical upgrade: cross-attention over the clique representation rather than flat conditioning. This lets the flow transformer "look at" different cliques when reconstructing different geometric aspects of the material. The classifier-free guidance trick (from the image generation literature — Ho & Salimans, 2022) sharpens the decoded structure.

**The Optimizer** uses evolution strategies rather than backpropagation — a lesson from the adversarial input problem that plagued every approach since Chen & Gu (2020). ES uses only rank ordering of candidates, not gradient magnitudes, making it robust to spurious features in the learned landscape. Weight decay pulls the latent toward the prior, providing implicit regularization against distributional shift — the same conservative instinct that motivated COMs, implemented through a simpler mechanism.

### The Results in Context

The numbers tell the story of a paradigm shift:

| Method | Type | Formation Energy ↓ | Band Gap ↓ |
|--------|------|-------------------|------------|
| CrystalFormer | Generative | 0.71 | 0.48 |
| DiffCSP | Generative | 0.59 | 0.63 |
| DiffCSP++ | Generative | 0.65 | 0.60 |
| MatterGen | Generative | 0.60 | 0.57 |
| **CliqueFlowmer** | **MBO** | **-0.81** | **0.03** |
| **CliqueFlowmer-Top** | **MBO** | **-0.99** | **0.03** |

The generative baselines cluster around 0.5–0.7 for both tasks — they're sampling from approximately the same distribution, so they produce approximately the same property statistics. CliqueFlowmer doesn't just improve on them; it operates in a qualitatively different regime. Negative formation energy means the optimized materials are predicted to be *more stable than their raw elements* — a physically meaningful threshold that no generative baseline consistently crosses.

For band gap, CliqueFlowmer drives values to near zero — metallic or near-metallic behavior — while baselines spread across the full range. The model isn't sampling and hoping; it's systematically finding the materials with the lowest band gap in the latent space.

---

## Chapter 5: The Parallel Paths Not Taken

### GFlowNets: Proportional Sampling Instead of Optimization

**Bengio et al. JMLR 2023; Hernandez-Garcia et al. NeurIPS 2023 Workshop (Crystal-GFN).**

While CliqueFlowmer optimizes for the single best material, GFlowNets (Generative Flow Networks) take a philosophically different approach: sample materials *proportionally to a reward function*. If the reward is $R(M) = \exp(-\beta \cdot E_\text{form}(M))$, GFlowNets generate materials with probability proportional to $R(M)$ — concentrating on low-energy materials without collapsing to a single mode.

Crystal-GFN applied this to crystal structures, sequentially sampling space group → composition → lattice parameters. It found diverse crystals with low formation energy (median -3.2 eV/atom) while maintaining chemical variety.

The GFlowNet approach has a compelling advantage: **diversity**. By sampling proportionally rather than optimizing greedily, GFlowNets naturally produce a diverse set of high-quality candidates. This matters in practice because synthesis constraints, cost, and toxicity often make the "globally optimal" material impractical — you want a portfolio of good options.

The disadvantage is that GFlowNets are harder to train (the flow-consistency condition is tricky to enforce) and may not reach the extreme tails of the property distribution as aggressively as direct optimization.

**Future convergence**: A natural next step is combining GFlowNet-style diversity with CliqueFlowmer-style latent optimization — perhaps using GFlowNets to sample from a clique-decomposed energy function in latent space, getting both diversity and compositional generalization.

### Bayesian Optimization: The Classical Alternative

Traditional Bayesian optimization (BO) uses a Gaussian process surrogate and an acquisition function (expected improvement, upper confidence bound) to sequentially select experiments. BO is the gold standard for *online* optimization with expensive evaluations.

But BO struggles with CMD for the same reasons it struggles with any high-dimensional, structured input space: Gaussian processes scale poorly beyond ~20 dimensions, and they can't natively handle the mixed discrete-continuous, variable-dimensional structure of crystals. There's active research on high-dimensional BO (random embeddings, trust regions) and structured BO (graph kernels, molecular kernels), but none have been convincingly demonstrated on full crystal structure spaces.

CliqueFlowmer's latent space could be a natural target for BO — the clique structure provides a fixed-dimensional, continuous space where GP surrogates might work well. This is an unexplored direction.

### Active Learning and Closed-Loop Discovery

A complementary paradigm is closed-loop discovery: generate candidates → simulate (DFT) or synthesize → update the model → repeat. This is how programs like the A-Lab at Lawrence Berkeley National Laboratory operate — an autonomous laboratory that uses ML to propose materials, robotically synthesizes them, characterizes them, and feeds results back to the model.

CliqueFlowmer is currently a single-shot, offline method. But its latent optimization framework is naturally compatible with active learning: generate optimized candidates, evaluate the best ones with DFT, add them to the training set, retrain, and repeat. Each cycle would push the frontier of known materials into previously unexplored regions of property space.

---

## Chapter 6: The Deep Structure of Why This Is Hard

### The Thermodynamic Landscape

Understanding why CMD is hard requires understanding the *energy landscape* of materials.

Every arrangement of atoms has an energy, determined by quantum mechanics. The set of all possible arrangements and their energies defines a hypersurface — the potential energy surface (PES). Stable materials are local minima of this surface. The *global* minimum for a given composition defines the thermodynamic ground state.

But the PES is a nightmare landscape. It has:
- **Exponentially many local minima**: a unit cell with $N$ atoms has a PES with roughly $\exp(N)$ local minima (polymorphs)
- **Narrow funnels**: the basin of attraction around each minimum can be very narrow in some directions (soft modes) and very wide in others (hard modes)
- **Saddle points everywhere**: most critical points are saddles, not minima — the landscape is more like a mountain pass than a valley
- **Composition-dependent topology**: adding or removing one atom completely changes the PES topology

This means that even with a perfect energy model, *finding* the global minimum is NP-hard in general. Nature "solves" this through thermodynamic annealing — materials crystallize by cooling, exploring the landscape through thermal fluctuations. Simulated annealing and molecular dynamics mimic this process, but they're painfully slow for complex unit cells.

This is where ML methods have their biggest potential advantage: they can learn *shortcuts* through the landscape. Instead of physically simulating the annealing process, they learn correlations between structure and stability from data, enabling direct jumps to promising regions.

### The Stability Problem

Finding a local minimum of the PES isn't enough. A material is only synthesizable if it's *thermodynamically stable* — i.e., it sits on the *convex hull* of the energy-composition space.

The convex hull is constructed by plotting formation energy vs. composition for all known materials. The lower envelope (the hull) represents the set of compositions that are thermodynamically accessible. A material above the hull will decompose into a mixture of hull materials. The distance above the hull ($E_\text{hull}$) quantifies instability.

The Materials Project uses a threshold of $E_\text{hull} < 0.1$ eV/atom for "likely synthesizable" materials. This is a pragmatic choice — it accounts for the fact that some metastable materials can be kinetically trapped (diamond is metastable relative to graphite, but it persists because the barrier to conversion is enormous).

For CMD, this means that optimizing a target property (band gap, conductivity) is *constrained optimization*: you want to minimize band gap *subject to* the material being on or near the convex hull. This constraint makes the problem dramatically harder because the feasible region (near the hull) is a thin, convoluted surface in the vast space of all possible materials.

CliqueFlowmer handles this through its training data — the MP-20 materials are (mostly) on or near the hull, so the latent space naturally concentrates on feasible regions. But aggressive optimization can push latent codes into regions where the decoded materials drift off the hull. The weight decay regularizer mitigates this by anchoring the optimization near the data distribution.

### Why Transdimensionality Is the Real Bottleneck

Most discussions of CMD difficulty focus on the size of the search space. But the deeper issue is *transdimensionality* — the fact that different materials live in different-dimensional spaces.

Consider: a binary compound AB₂ with 3 atoms per cell lives in $\mathbb{R}^{6+9} = \mathbb{R}^{15}$ (6 lattice parameters + 9 fractional coordinates). A ternary compound A₂B₃C with 6 atoms per cell lives in $\mathbb{R}^{6+18} = \mathbb{R}^{24}$. These aren't just different points in the same space — they're points in *different spaces*.

Standard optimization methods can't handle this. Gradient descent needs a fixed-dimensional parameter vector. Evolutionary algorithms need crossover operators that work between solutions of the same dimensionality. Bayesian optimization needs a fixed-dimensional input to its surrogate.

CliqueFlowmer's encoder solves this by projecting all materials — regardless of atom count — into the same 121-dimensional latent space through attention-based pooling. This is a *dimensionality-collapsing* operation that throws away the variable-dimensional structure in favor of a fixed representation. The information about atom count is preserved implicitly in the latent code (the decoder can reconstruct it), but the optimizer works in a uniform space.

This is the same strategy that word embeddings use for natural language (variable-length sentences → fixed-dimensional vectors) and that protein language models use for biology (variable-length amino acid sequences → fixed-dimensional representations). The principle is universal: **embed the variable-dimensional object in a fixed-dimensional space where optimization is tractable, then decode back.**

---

## Chapter 7: Open Frontiers

### Better Oracles

CliqueFlowmer's results are bottlenecked by its oracles — M3GNet for formation energy and MEGNet for band gap. These are fast but imperfect: the paper shows that S.U.N. rates drop significantly when DFT replaces M3GNet for stability evaluation. Better ML oracles — or adaptive oracle refinement during optimization — could dramatically improve the quality of generated materials.

The frontier here is *universal* property predictors that cover not just energy and band gap, but catalytic activity, ionic conductivity, thermal conductivity, hardness, and other properties that currently lack cheap, accurate surrogates. Models like MACE-MP-0 and the newest universal potentials are moving in this direction.

### Multi-Objective Optimization

Real materials design is multi-objective: you want low cost *and* high conductivity *and* mechanical durability *and* environmental safety. CliqueFlowmer currently optimizes a single scalar property. Extending it to Pareto optimization over multiple objectives — finding the set of materials that are not dominated on any property — is a natural and important next step.

The clique decomposition might naturally support this: different cliques could be associated with different objectives, enabling targeted trade-offs.

### Active Learning Loops

The biggest near-term impact would come from closing the loop: CliqueFlowmer generates candidates → DFT evaluates the best → results are added to the training set → retrain → repeat. Each cycle pushes the Pareto frontier forward. The key question is how to select which candidates to evaluate — this is where ideas from Bayesian optimization (acquisition functions, information-theoretic approaches) could complement CliqueFlowmer's latent optimization.

### Beyond Crystals

The CliqueFlowmer architecture — VAE encoder with attention pooling + clique-decomposed predictor + autoregressive discrete decoder + flow-based continuous decoder — is not inherently limited to crystals. The same framework could apply to:

- **Molecules and drug design**: variable numbers of atoms, mixed discrete-continuous structure, property optimization
- **Proteins**: variable-length sequences, 3D structure, fitness landscape optimization
- **Metamaterials**: variable topology, mechanical/optical property optimization
- **Polymers**: variable monomer sequences, material property optimization

The principle is the same everywhere: embed variable-structure objects in a clique-decomposed latent space, optimize there, decode back. The specific encoder and decoder architectures change, but the MBO core transfers.

### Learning the Clique Structure

Currently, the clique decomposition (number of cliques, clique dimension, knot size, chain topology) is fixed by hyperparameter choices. An exciting theoretical direction is *learning the decomposition itself* — using structure learning algorithms to discover the optimal factorization of the objective function.

This connects to a deep question in machine learning: what is the right modular structure for a given problem? The answer might depend on the target property (band gap might decompose differently from formation energy) and could be discovered through meta-learning or neural architecture search.

### Scaling Laws for Materials

The LLM revolution was partly driven by scaling laws — predictable relationships between model size, data size, and performance. Do similar scaling laws exist for materials models? If doubling the training data predictably improves the best achievable property value, that would provide a roadmap for data collection priorities.

Early evidence suggests that materials prediction models do exhibit scaling behavior, but the picture is complicated by the heterogeneity of materials space (adding more oxide data doesn't help predict nitride properties). Understanding these scaling relationships could guide the allocation of computational resources between model training and DFT evaluation.

---

## Epilogue: From Sampling to Searching

The history of AI for materials discovery is a story of the field slowly learning that **generation is not the same as optimization**.

The first generation of methods (CGCNN, MEGNet, M3GNet) learned to *see* materials — to predict their properties from structure. The second generation (CDVAE, DiffCSP, MatterGen) learned to *dream* materials — to generate new structures that look like known ones. The third generation — CliqueFlowmer — learned to *search* for materials — to systematically find structures that optimize a target property.

Each generation built on the last. You need property predictors (Thread 1) to evaluate candidates. You need generative architectures (Thread 2) to encode and decode crystal structures. And you need optimization theory (Thread 3) to navigate the search space efficiently. CliqueFlowmer is the first method to weave all three threads together.

The deeper story is about the right way to formulate the problem. For years, the field implicitly framed CMD as a *density estimation* problem: learn the distribution of materials, sample from it. This framing was natural — it's what generative models do — but it was mismatched to the goal. Materials scientists don't want *typical* materials; they want *exceptional* ones.

CliqueFlowmer reframes CMD as what it always was: a *constrained optimization* problem. The constraint is physical realizability (the material must be stable and synthesizable). The objective is the target property. The data is the Materials Project. And the method is offline model-based optimization with compositional structure.

The library of $10^{60}$ books is still infinite. But now, instead of wandering the shelves and hoping to stumble on something good, we have a map — imperfect, incomplete, but getting better — that points toward the books most worth reading.

---

## Key References (Chronological)

| Year | Paper | Core Contribution |
|------|-------|-------------------|
| 2013 | Materials Project (Jain et al.) | The foundational database — 150K+ computed materials |
| 2017 | SchNet (Schütt et al.) | Continuous-filter convolutions for atomistic GNNs |
| 2017 | Transformer (Vaswani et al.) | Self-attention architecture underlying modern models |
| 2018 | CGCNN (Xie & Grossman) | First GNN for periodic crystal property prediction |
| 2019 | MEGNet (Chen et al.) | Global-state graph networks for materials |
| 2020 | DDPM (Ho, Jain, Abbeel) | Denoising diffusion for generation |
| 2020 | MINs (Kumar & Levine) | Model inversion for offline optimization |
| 2021 | COMs (Trabucco et al.) | Conservative surrogates prevent adversarial drift |
| 2021 | ALIGNN (Choudhary & DeCost) | Line graph GNN captures angular information |
| 2022 | CDVAE (Xie et al.) | First complete crystal generative model; MP-20 benchmark |
| 2022 | M3GNet (Chen & Ong) | Universal interatomic potential for 89 elements |
| 2022 | Design-Bench (Trabucco et al.) | Standardized MBO benchmarks |
| 2022 | Flow Matching (Lipman et al.) | Simple, scalable training for continuous normalizing flows |
| 2023 | DiffCSP (Jiao et al.) | Joint diffusion on lattice + fractional coordinates |
| 2023 | GFlowNets (Bengio et al.) | Proportional reward sampling; Crystal-GFN variant |
| 2023 | DiT (Peebles & Xie) | AdaLN conditioning for diffusion transformers |
| 2024 | DiffCSP++ (Jiao et al.) | Space group symmetry constraints |
| 2024 | FlowMM (Miller et al.) | Riemannian flow matching for crystals |
| 2024 | Functional Graphical Models (Grudzien et al.) | Theory of clique-decomposed optimization |
| 2024 | Cliqueformer (Kuba et al.) | Clique-structured MBO achieves SOTA on Design-Bench |
| 2025 | MatterGen (Zeni et al.) | Industrial-scale crystal generation; Nature publication |
| 2025 | CrystalFormer-Gen (Cao et al.) | Autoregressive transformer crystal generation |
| **2026** | **CliqueFlowmer (Kuba et al.)** | **First MBO method for CMD; converges all three threads** |

---

*Document created for Rabbit Hole-a-thon research, March 2026.*
