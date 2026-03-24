from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch

from pymatgen.core import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.structure_matcher import StructureMatcher

from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry


from data import tools
from data.constants import atomic_symbols
import warnings
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class RefEntry:
    split: str          # "train" / "val" / "test"
    idx: int            # index within that split
    composition: Composition
    element_set: frozenset  # set of element symbols (for novelty candidate filtering)
    energy: float       # formation energy (from targets or oracle)


def _targets_to_float_list(x) -> List[float]:
    """
    Normalize targets (could be numpy, list, or torch.Tensor on CPU/GPU) into a flat list of floats.
    """
    if isinstance(x, torch.Tensor):
        return (
            x.detach()
             .cpu()
             .view(-1)
             .to(torch.float32)
             .tolist()
        )
    # numpy array, list, etc.
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr.tolist()


def _composition_from_atomic_mask(
    atomic: torch.Tensor,
    mask: torch.Tensor,
) -> List[Composition]:
    """
    atomic: (B, L) integer Z indices
    mask:   (B, L) {0,1}
    Returns: list of pymatgen Composition for each batch element.
    Assumes first and last masked tokens are BOS/EOS and skips them.
    """
    atomic = atomic.to("cpu")
    mask = mask.to("cpu")

    B, L = atomic.shape
    out: List[Composition] = []

    for b in range(B):
        m = mask[b].bool()
        if m.sum().item() == 0:
            out.append(Composition())
            continue

        # positions with tokens (including BOS/EOS)
        idx_all = torch.where(m)[0]
        if len(idx_all) <= 2:
            out.append(Composition())
            continue

        # drop first and last (BOS/EOS)
        idx_valid = idx_all[1:-1]
        counts: Dict[str, int] = {}
        for j in idx_valid:
            z = int(atomic[b, j].item())
            sym = atomic_symbols[z]  # your existing mapping
            counts[sym] = counts.get(sym, 0) + 1

        out.append(Composition(counts))

    return out


def _compositions_for_inputs(
    inputs,
    split_name: str,
    batch_size: int,
    device: torch.device,
    energies: List[float],
    record_freq : int = 100
) -> List[RefEntry]:
    """
    Build RefEntry list (composition + energy) for one split, without constructing full Structures.
    """
    entries: List[RefEntry] = []
    n = len(inputs)
    energy_arr = np.asarray(energies, dtype=float).reshape(-1)
    assert n == len(energy_arr)

    global_idx = 0
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_raw = [inputs[i] for i in range(start, end)]
        abc, angles, atomic, pos, mask = tools.move_to_device(
            tools.unpack_structures(batch_raw), device
        )
        comps = _composition_from_atomic_mask(atomic, mask)

        for i_local, comp in enumerate(comps):
            e_form = float(energy_arr[start + i_local])
            elemset = frozenset(el.symbol for el in comp.elements)
            entries.append(
                RefEntry(
                    split=split_name,
                    idx=start + i_local,
                    composition=comp,
                    element_set=elemset,
                    energy=e_form,
                )
            )
        global_idx += (end - start)

        if record_freq > 0 and start % record_freq == 0:
            print(f"Composition for inputs: recorded {(start + 1) * batch_size} entries.")

    return entries


def build_reference_metadata(
    train_inputs,
    val_inputs,
    test_inputs,
    train_targets,
    val_targets,
    test_targets,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
    hull_splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Tuple[List[RefEntry], Dict[frozenset, List[RefEntry]]]:
    """
    Build:
      - ref_entries: list of RefEntry (composition + energy, no Structures)
      - by_elemset: mapping element_set -> list[RefEntry] for novelty/hull subsets
    """

    ref_entries: List[RefEntry] = []

    if "train" in hull_splits:
        train_e = _targets_to_float_list(train_targets)
        ref_entries.extend(
            _compositions_for_inputs(
                train_inputs,
                split_name="train",
                batch_size=batch_size,
                device=device,
                energies=train_e,
            )
        )

    if "val" in hull_splits and val_inputs is not None:
        val_e = _targets_to_float_list(val_targets)
        ref_entries.extend(
            _compositions_for_inputs(
                val_inputs,
                split_name="val",
                batch_size=batch_size,
                device=device,
                energies=val_e,
            )
        )

    if "test" in hull_splits and test_inputs is not None:
        test_e = _targets_to_float_list(test_targets)
        ref_entries.extend(
            _compositions_for_inputs(
                test_inputs,
                split_name="test",
                batch_size=batch_size,
                device=device,
                energies=test_e,
            )
        )

    by_elemset: Dict[frozenset, List[RefEntry]] = {}
    for e in ref_entries:
        by_elemset.setdefault(e.element_set, []).append(e)

    return ref_entries, by_elemset


_ref_struct_cache: Dict[Tuple[str, int], Structure] = {}


def _load_structure_from_entry(
    entry: RefEntry,
    train_inputs,
    val_inputs,
    test_inputs,
    device: torch.device = torch.device("cpu"),
) -> Structure:
    """
    Lazily decode a single MP-20 entry into a pymatgen.Structure, using your existing tools.
    Uses a small in-memory cache to avoid decoding the same entry multiple times.
    """
    key = (entry.split, entry.idx)
    if key in _ref_struct_cache:
        return _ref_struct_cache[key]

    if entry.split == "train":
        raw = [train_inputs[entry.idx]]
    elif entry.split == "val":
        raw = [val_inputs[entry.idx]]
    elif entry.split == "test":
        raw = [test_inputs[entry.idx]]
    else:
        raise ValueError(f"Unknown split {entry.split}")

    abc, angles, atomic, pos, mask = tools.move_to_device(
        tools.unpack_structures(raw), device
    )
    struct = tools.tensors_to_structure(abc, angles, atomic, pos, mask)[0]
    _ref_struct_cache[key] = struct
    return struct


def structurally_valid(struct: Structure, min_dist: float = 0.5) -> bool:
    dists = struct.distance_matrix
    dists = dists + np.eye(len(dists)) * 1e9
    return (dists > min_dist).all()


def compositionally_valid(struct: Structure) -> bool:
    # SMACT or similar can be plugged here; accept all for now
    return True


def compute_ehull_for_structure(
    struct: Structure,
    oracle,
    phase_diagram: PhaseDiagram,
) -> float:
    """
    Compute Ehull (eV/atom) for one structure using oracle energies against the reference hull.
    """
    e_form = float(oracle.predict_structure(struct))
    entry = PDEntry(struct.composition, e_form)
    ehull = phase_diagram.get_e_above_hull(entry)
    return float(ehull)


def compute_local_ehull_for_structure(
    struct: Structure,
    oracle,
    ref_entries: List[RefEntry],
) -> float:
    """
    Compute Ehull (eV/atom) for a single structure by building a local PhaseDiagram
    restricted to the element subset of 'struct'.

    This avoids a single huge high-D hull over all MP-20 elements.
    """
    # energy of candidate
    e_form = float(oracle.predict_structure(struct))
    comp_c = struct.composition
    elemset_c = frozenset(el.symbol for el in comp_c.elements)

    # need at least 2 distinct elements for a meaningful hull
    if len(elemset_c) < 1:
        return float("inf")

    # restrict reference entries to this chemical subspace
    sub_entries = [
        e for e in ref_entries
        if len(e.element_set) > 0 and e.element_set.issubset(elemset_c)
    ]

    # if too few references, hull is ill-defined; treat as unstable
    if len(sub_entries) < 1:
        return float("inf")

    pd_entries = [PDEntry(e.composition, e.energy) for e in sub_entries]
    cand_entry = PDEntry(comp_c, e_form)

    # build local PD
    pd = PhaseDiagram(pd_entries + [cand_entry])
    ehull = pd.get_e_above_hull(cand_entry)
    return float(ehull)


def load_matbench_ppd(ppd_path: str) -> PatchedPhaseDiagram:
    obj = loadfn(ppd_path)
    if isinstance(obj, PatchedPhaseDiagram):
        return obj
    if isinstance(obj, dict) and obj.get("@class") == "PatchedPhaseDiagram":
        return PatchedPhaseDiagram.from_dict(obj)
    raise TypeError(f"Unexpected object in {ppd_path}: {type(obj)}")


def compute_ppd_ehull_from_m3gnet(struct, oracle, ppd) -> float:
    """
    Candidate energy: M3GNet formation energy PER ATOM (oracle.predict_structure)
    PPD expects entry.energy to be TOTAL energy for the composition.
    """
    e_form_per_atom = float(oracle.predict_structure(struct))
    n_atoms = float(struct.composition.num_atoms)
    e_total = e_form_per_atom * n_atoms

    entry = ComputedEntry(struct.composition, e_total)
    eh = ppd.get_e_above_hull(entry)

    eh = float(eh)
    if not np.isfinite(eh):
        return float("inf")
    return eh


def classify_sun_for_optimized(
    optimized_structs: List[Structure],    # refined primitives from your pipeline
    train_inputs,
    val_inputs,
    test_inputs,
    ref_entries: List["RefEntry"],
    by_elemset: Dict[frozenset, List["RefEntry"]],
    oracle,
    stable_threshold: float = 0.0,
    metastable_threshold: float = 0.1,
    min_dist: float = 0.5,
    ref_device: torch.device = torch.device("cpu"),
) -> Tuple[List[int], Dict[str, object]]:
    matcher = StructureMatcher()

    N_total = len(optimized_structs)

    # 1) Validity filter (keep mapping to original indices)
    valid_pairs = [
        (orig_i, s)
        for orig_i, s in enumerate(optimized_structs)
        if structurally_valid(s, min_dist=min_dist) and compositionally_valid(s)
    ]
    valid_indices = [i for i, _ in valid_pairs]          # original indices
    valid_structs = [s for _, s in valid_pairs]
    N_gen = len(valid_structs)

    print("ELEMSET FILTER | kept", N_gen, "of", N_total)

    # 2) Ehull + element sets (for valid structs only)
    ehulls = []
    elemsets = []
    for s in valid_structs:
        eh = compute_local_ehull_for_structure(s, oracle, ref_entries)
        ehulls.append(eh)
        elemsets.append(frozenset(el.symbol for el in s.composition.elements))

    ehulls = np.asarray(ehulls, dtype=float)
    ehulls[~np.isfinite(ehulls)] = float("inf")

    print(
        "EHULL DIAG",
        "min", float(np.min(ehulls)) if N_gen > 0 else float("nan"),
        "p1",  float(np.quantile(ehulls, 0.01)) if N_gen > 0 else float("nan"),
        "p5",  float(np.quantile(ehulls, 0.05)) if N_gen > 0 else float("nan"),
        "p50", float(np.quantile(ehulls, 0.50)) if N_gen > 0 else float("nan"),
        "p95", float(np.quantile(ehulls, 0.95)) if N_gen > 0 else float("nan"),
        "finite_frac", float(np.isfinite(ehulls).mean()) if N_gen > 0 else 0.0,
    )

    multi_elem = np.array([len(es) >= 2 for es in elemsets], dtype=bool)
    stable_mask = (ehulls <= stable_threshold) & multi_elem
    meta_mask   = (ehulls <= metastable_threshold) & multi_elem

    # 3) Novelty vs MP-20 refs (valid structs only)
    novelty_mask = np.ones(N_gen, dtype=bool)
    for i, (s, es) in enumerate(zip(valid_structs, elemsets)):
        ref_candidates = by_elemset.get(es, [])
        for entry in ref_candidates:
            ref_struct = _load_structure_from_entry(
                entry,
                train_inputs=train_inputs,
                val_inputs=val_inputs,
                test_inputs=test_inputs,
                device=ref_device,
            )
            if matcher.fit(s, ref_struct):
                novelty_mask[i] = False
                break

    # --- NEW: report base rates (stability / novelty / uniqueness) for "generated valid" population ---
    # Stability rate here = fraction of valid generated that are stable/metastable (with >=2 elements, consistent with masks).
    stability_rate_stable = float(stable_mask.mean()) if N_gen > 0 else 0.0
    stability_rate_meta   = float(meta_mask.mean()) if N_gen > 0 else 0.0

    # Novelty rate = fraction of valid generated that are novel vs MP-20 refs (under StructureMatcher).
    novelty_rate_all = float(novelty_mask.mean()) if N_gen > 0 else 0.0

    # Uniqueness rate = fraction of valid generated that are unique among themselves (StructureMatcher clustering).
    # We compute uniqueness on *all valid_structs* (not only SUN) to match "uniqueness rate" as a generic generative metric.
    def _unique_reps_indices(structs: List[Structure]) -> List[int]:
        reps_idx: List[int] = []
        used = [False] * len(structs)
        for i, s in enumerate(structs):
            if used[i]:
                continue
            used[i] = True
            reps_idx.append(i)
            for j in range(i + 1, len(structs)):
                if not used[j] and matcher.fit(s, structs[j]):
                    used[j] = True
        return reps_idx

    unique_all_idx_valid = _unique_reps_indices(valid_structs)
    uniqueness_rate_all = (len(unique_all_idx_valid) / N_gen) if N_gen > 0 else 0.0

    # 4) Build sets (for SUN)
    stable_structs = [valid_structs[i] for i in range(N_gen) if stable_mask[i]]
    stable_novel   = [valid_structs[i] for i in range(N_gen) if stable_mask[i] and novelty_mask[i]]

    meta_structs = [valid_structs[i] for i in range(N_gen) if meta_mask[i]]
    meta_novel   = [valid_structs[i] for i in range(N_gen) if meta_mask[i] and novelty_mask[i]]

    # 5) Uniqueness for SUN pools (return reps + their indices in valid_structs)
    def unique_reps_with_idx(structs: List[Structure], idxs: List[int]) -> Tuple[List[Structure], List[int]]:
        reps: List[Structure] = []
        rep_idxs: List[int] = []
        used = [False] * len(structs)
        for i, s in enumerate(structs):
            if used[i]:
                continue
            used[i] = True
            reps.append(s)
            rep_idxs.append(idxs[i])  # index in valid_structs
            for j in range(i + 1, len(structs)):
                if not used[j] and matcher.fit(s, structs[j]):
                    used[j] = True
        return reps, rep_idxs

    stable_novel_idxs_valid = [i for i in range(N_gen) if stable_mask[i] and novelty_mask[i]]
    meta_novel_idxs_valid   = [i for i in range(N_gen) if meta_mask[i] and novelty_mask[i]]

    stable_unique, stable_unique_idxs_valid = unique_reps_with_idx(
        [valid_structs[i] for i in stable_novel_idxs_valid],
        stable_novel_idxs_valid,
    )
    meta_unique, meta_unique_idxs_valid = unique_reps_with_idx(
        [valid_structs[i] for i in meta_novel_idxs_valid],
        meta_novel_idxs_valid,
    )

    # --- Sort SUN reps by Ehull (lowest first) ---
    stable_pairs = sorted(
        zip(stable_unique_idxs_valid, stable_unique),
        key=lambda p: float(ehulls[p[0]])
    )
    stable_unique_idxs_valid = [i for i, _ in stable_pairs]
    stable_unique = [s for _, s in stable_pairs]

    meta_pairs = sorted(
        zip(meta_unique_idxs_valid, meta_unique),
        key=lambda p: float(ehulls[p[0]])
    )
    meta_unique_idxs_valid = [i for i, _ in meta_pairs]
    meta_unique = [s for _, s in meta_pairs]

    # Convert "valid_structs indices" -> "optimized_structs original indices"
    sun_stable_original_idxs = [valid_indices[i] for i in stable_unique_idxs_valid]
    sun_meta_original_idxs   = [valid_indices[i] for i in meta_unique_idxs_valid]

    # --- build ALL original indices list:
    # output_idxs_ascending_ehull = SUN-stable (already sorted by Ehull) + all remaining (sorted by Ehull)
    all_ehulls = np.full(N_total, float("inf"), dtype=float)
    for valid_pos, orig_i in enumerate(valid_indices):
        all_ehulls[orig_i] = float(ehulls[valid_pos])

    sun_set = set(sun_stable_original_idxs)
    other_idxs_ascending_ehull = sorted(
        [i for i in range(N_total) if i not in sun_set],
        key=lambda i: (float(all_ehulls[i]), int(i)),
    )
    output_idxs_ascending_ehull = sun_stable_original_idxs + other_idxs_ascending_ehull

    N_stable       = len(stable_structs)
    N_meta         = len(meta_structs)
    N_SUN_stable   = len(stable_unique)
    N_SUN_meta     = len(meta_unique)

    sun_rate_stable = N_SUN_stable / N_gen if N_gen > 0 else 0.0
    sun_rate_meta   = N_SUN_meta   / N_gen if N_gen > 0 else 0.0

    # --- NEW: novelty/uniqueness rates also for stable/metastable subsets (optional but useful) ---
    # Novelty among stable/metastable (fraction within those subsets that are novel)
    stable_novelty_rate = (
        float(novelty_mask[stable_mask].mean()) if np.any(stable_mask) else 0.0
    )
    meta_novelty_rate = (
        float(novelty_mask[meta_mask].mean()) if np.any(meta_mask) else 0.0
    )

    # Uniqueness among stable/metastable pools (computed as unique clusters / pool size)
    stable_unique_rate_pool = (N_SUN_stable / len(stable_novel_idxs_valid)) if len(stable_novel_idxs_valid) > 0 else 0.0
    meta_unique_rate_pool   = (N_SUN_meta   / len(meta_novel_idxs_valid))   if len(meta_novel_idxs_valid) > 0 else 0.0

    # Print the new rates in a compact, paper-friendly way
    print(
        "SUN RATES | "
        f"stability(stable)={stability_rate_stable:.3f} "
        f"stability(meta)={stability_rate_meta:.3f} "
        f"novelty(all)={novelty_rate_all:.3f} "
        f"uniqueness(all)={uniqueness_rate_all:.3f} "
        f"SUN(stable)={sun_rate_stable:.3f} "
        f"SUN(meta)={sun_rate_meta:.3f} "
        f"novelty(stable)={stable_novelty_rate:.3f} "
        f"novelty(meta)={meta_novelty_rate:.3f} "
        f"uniq_in_pool(stable)={stable_unique_rate_pool:.3f} "
        f"uniq_in_pool(meta)={meta_unique_rate_pool:.3f}"
    )

    out = {
        "N_total": N_total,
        "N_gen": N_gen,

        # Ehull for valid structs (aligned with valid_structs), plus Ehull for all originals (inf for invalid)
        "ehulls_valid": ehulls,
        "ehulls_all": all_ehulls,
        "valid_indices": valid_indices,

        "N_stable": N_stable,
        "N_SUN_stable": N_SUN_stable,
        "StabilityRate_stable": float(stability_rate_stable),
        "SUNRate_stable": float(sun_rate_stable),
        "stable_structs": stable_structs,
        "SUN_structs_stable": stable_unique,
        "SUN_indices_stable": sun_stable_original_idxs,

        "N_metastable": N_meta,
        "N_SUN_metastable": N_SUN_meta,
        "StabilityRate_metastable": float(stability_rate_meta),
        "SUNRate_metastable": float(sun_rate_meta),
        "metastable_structs": meta_structs,
        "SUN_structs_metastable": meta_unique,
        "SUN_indices_metastable": sun_meta_original_idxs,

        # --- NEW: global “gen-quality” rates on valid generated population ---
        "NoveltyRate_all": float(novelty_rate_all),
        "UniquenessRate_all": float(uniqueness_rate_all),
        "unique_indices_valid_all": unique_all_idx_valid,  # indices into valid_structs (not originals)

        # --- NEW: conditional rates (often used in papers) ---
        "NoveltyRate_stable": float(stable_novelty_rate),
        "NoveltyRate_metastable": float(meta_novelty_rate),
        "UniquenessRate_in_pool_stable_novel": float(stable_unique_rate_pool),
        "UniquenessRate_in_pool_metastable_novel": float(meta_unique_rate_pool),

        # New primary ordering output
        "output_indices_ascending_ehull": output_idxs_ascending_ehull,
        "other_indices_ascending_ehull": other_idxs_ascending_ehull,
    }

    # Return the concatenated list (SUN stable first; all remaining next, sorted by Ehull), plus dict.
    return output_idxs_ascending_ehull, out



def classify_sun_with_ehull_fn(
    optimized_structs: List[Structure],
    train_inputs,
    val_inputs,
    test_inputs,
    by_elemset,
    ehull_fn,
    *,
    stable_threshold: float = 0.0,
    metastable_threshold: float = 0.08,
    min_dist: float = 0.5,
    ref_device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from optimization.sun import _load_structure_from_entry

    matcher = StructureMatcher()

    # 1) Validity filter
    min_ref_count = 20  # IMPORTANT: tuneable, but start here # 200,

    valid_structs = []
    for s in optimized_structs:
        if not structurally_valid(s, min_dist=min_dist):
            continue
        if not compositionally_valid(s):
            continue

        elemset = frozenset(el.symbol for el in s.composition.elements)

        # Require at least binary
        if len(elemset) < 2:
            continue

        # Require enough reference support
        if len(by_elemset.get(elemset, [])) < min_ref_count:
            continue

        valid_structs.append(s)

    print(
        "ELEMSET FILTER | kept",
        len(valid_structs),
        "of",
        len(optimized_structs),
    )

    N_total = len(optimized_structs)
    N_gen = len(valid_structs)

    ehulls = []
    elemsets = []
    for s in valid_structs:
        ehulls.append(float(ehull_fn(s)))
        elemsets.append(frozenset(el.symbol for el in s.composition.elements))

    ehulls = np.asarray(ehulls, dtype=float)

    print("EHULL DIAG",
        "min", float(np.min(ehulls)),
        "p1", float(np.quantile(ehulls, 0.01)),
        "p5", float(np.quantile(ehulls, 0.05)),
        "p50", float(np.quantile(ehulls, 0.50)),
        "p95", float(np.quantile(ehulls, 0.95)),
        "finite_frac", float(np.isfinite(ehulls).mean()))

    multi_elem = np.array([len(es) >= 2 for es in elemsets])
    stable_mask = (ehulls <= stable_threshold) & multi_elem
    meta_mask   = (ehulls <= metastable_threshold) & multi_elem

    novelty_mask = np.ones(N_gen, dtype=bool)
    for i, (s, es) in enumerate(zip(valid_structs, elemsets)):
        ref_candidates = by_elemset.get(es, [])
        for entry in ref_candidates:
            ref_struct = _load_structure_from_entry(
                entry,
                train_inputs=train_inputs,
                val_inputs=val_inputs,
                test_inputs=test_inputs,
                device=ref_device,
            )
            if matcher.fit(s, ref_struct):
                novelty_mask[i] = False
                break

    stable_novel = [valid_structs[i] for i in range(N_gen) if stable_mask[i] and novelty_mask[i]]
    meta_novel   = [valid_structs[i] for i in range(N_gen) if meta_mask[i] and novelty_mask[i]]

    def unique_reps(structs: List[Structure]) -> List[Structure]:
        reps = []
        used = [False] * len(structs)
        for i, s in enumerate(structs):
            if used[i]:
                continue
            used[i] = True
            reps.append(s)
            for j in range(i + 1, len(structs)):
                if not used[j] and matcher.fit(s, structs[j]):
                    used[j] = True
        return reps

    stable_unique = unique_reps(stable_novel)
    meta_unique   = unique_reps(meta_novel)

    N_stable = int(stable_mask.sum())
    N_meta   = int(meta_mask.sum())
    N_SUN_stable = len(stable_unique)
    N_SUN_meta   = len(meta_unique)

    return {
        "N_total": N_total,
        "N_gen": N_gen,
        "ValidityRate_structural": (N_gen / N_total) if N_total else 0.0,
        "ehulls": ehulls.tolist(),

        "N_stable": N_stable,
        "N_SUN_stable": N_SUN_stable,
        "StabilityRate_stable": (N_stable / N_gen) if N_gen else 0.0,
        "SUNRate_stable": (N_SUN_stable / N_gen) if N_gen else 0.0,

        "N_metastable": N_meta,
        "N_SUN_metastable": N_SUN_meta,
        "StabilityRate_metastable": (N_meta / N_gen) if N_gen else 0.0,
        "SUNRate_metastable": (N_SUN_meta / N_gen) if N_gen else 0.0,

        # keep structures if you want, but DO NOT json.dump them
        "SUN_structs_stable": stable_unique,
        "SUN_structs_metastable": meta_unique,
    }
