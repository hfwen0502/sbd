# Subspace Extension and Energy Variance

This document describes features added to the SBD diagonalization program:
1. **Singles+doubles subspace extension** (`--carryover_type 4/5/6`) — expand the determinant space by applying single and double excitations to dominant half-determinants
2. **ERI-screened subspace extension** (`--carryover_type 7/8`) — like S+D but filtered by Hamiltonian integral magnitude, producing smaller and higher-quality subspaces
3. **Variance-only mode** (`--iteration 0`) — compute energy variance without diagonalization, enabling variance extrapolation workflows

## New Carryover Types

### Brute-Force Singles+Doubles (Types 4–6)

Three carryover types extend the existing singles-only expansion (types 1–3) by adding **same-spin double excitations**. For a half-determinant with $n$ occupied and $m$ virtual orbitals, this generates $\binom{n}{2}\binom{m}{2}$ doubles in addition to $n \times m$ singles.

| Type | Selection | Extension | Description |
|------|-----------|-----------|-------------|
| **4** | Amplitude | Singles + Doubles | Like type 3, but includes same-spin double excitations |
| **5** | Marginal + Amplitude | Singles + Doubles | Like type 2, but with S+D extension |
| **6** | None | Singles + Doubles | Expand ALL dets via S+D (no amplitude filtering) |

**Choosing between types:**
- **Type 4** is the most practical for iterative refinement — it expands only from dets with significant amplitude, keeping the subspace manageable. Control growth with `--carryover_threshold`.
- **Type 6** expands all dets unconditionally. The resulting subspace can be very large (e.g., 30k+ dets from 275 for H2O). Use when the initial subspace is small.
- **Types 4–5** respect `--carryover_threshold`: a smaller threshold includes more dets in the expansion basis. Set to 0 to expand from all dets with any nonzero amplitude.

### ERI-Screened Singles+Doubles (Types 7–8)

Types 7 and 8 add **Hamiltonian-integral-based screening** to the S+D extension, inspired by the extended SQD method (Cleveland Clinic / SONIC project). Instead of generating all combinatorially possible excitations, only excitations with significant Hamiltonian coupling are kept:

- **Singles** i→a: kept if `|h_{ia}| + Σ_m |v_{ia,mm} - v_{im,ma}| + Σ_m |v_{ia,m'm'}| > eri_threshold`, where the sum runs over same-spin and opposite-spin occupied orbitals. This is effectively the Fock matrix element magnitude.
- **Doubles** i,j→a,b: kept if `|v_{ia,jb} - v_{ib,ja}| > eri_threshold`, i.e., the antisymmetrized two-electron integral exceeds the threshold.

| Type | Selection | Extension | Description |
|------|-----------|-----------|-------------|
| **7** | Amplitude | Screened S+D | Like type 4, but with ERI screening |
| **8** | None | Screened S+D | Like type 6, but with ERI screening |

**New CLI option:**
- `--eri_threshold <float>`: Screening threshold for Hamiltonian integrals (default: 1e-6). Smaller values keep more excitations; larger values are more aggressive.

**Why ERI screening matters:**

For a 29-orbital system with 5 electrons per spin, brute-force S+D generates ~2880 excitations per half-determinant, but many have negligible Hamiltonian coupling (the corresponding `v_{ia,jb}` integral is near zero). ERI screening typically keeps 20-50% of excitations while retaining all physically important ones — the excitations that actually contribute to energy lowering.

**Choosing `--eri_threshold`:**
- **1e-8**: very mild screening, keeps almost all excitations
- **1e-6** (default): moderate screening, good balance of subspace size vs quality
- **1e-4**: aggressive screening, produces compact subspaces
- **1e-3**: very aggressive, may miss important excitations

```bash
# ERI-screened S+D with amplitude filtering
mpirun -np $NP ./diag \
    --fcidump fcidump.txt \
    --adetfile alpha_dets.txt \
    --iteration 300 \
    --carryover_type 7 \
    --carryover_threshold 0.001 \
    --eri_threshold 1e-6 \
    --savename wf_ \
    --carryover_adetfile expanded_alpha.txt

# ERI-screened S+D on ALL dets
mpirun -np $NP ./diag \
    --fcidump fcidump.txt \
    --adetfile alpha_dets.txt \
    --iteration 300 \
    --carryover_type 8 \
    --eri_threshold 1e-4 \
    --savename wf_ \
    --carryover_adetfile expanded_alpha.txt
```

## Variance-Only Mode (`--iteration 0`)

Setting `--iteration 0` skips the Davidson/Lanczos diagonalization entirely. Instead, it:
1. Loads a pre-computed wavefunction via `--loadname`
2. Builds the Hamiltonian diagonal terms
3. Computes one matrix-vector product $C = H|\psi\rangle$
4. Reports the wavefunction norm $\|\psi\|^2$, energy $E = \langle\psi|H|\psi\rangle / \|\psi\|^2$, and variance $\sigma^2 = \langle H\psi|H\psi\rangle / \|\psi\|^2 - E^2$

This measures how well a wavefunction approximates an eigenstate **in a larger space** than where it was optimized. If $|\psi\rangle$ is an eigenvector of the projected Hamiltonian in subspace $S$, the variance in $S$ is zero. But evaluated in a larger space $S' \supset S$, the variance captures $H$-connected components outside $S$.

## Variance Extrapolation Workflow

By running a sequence of increasing subspace sizes, each producing an (energy, variance) pair, one can extrapolate the energy toward the zero-variance limit (exact eigenvalue).

### Two-Step Protocol

Each iteration consists of two SBD runs:

**Step 1 — Diagonalize and expand:**
```bash
mpirun -np $NP ./diag \
    --fcidump fcidump.txt \
    --adetfile current_alpha.txt \
    --iteration 300 \
    --carryover_type 4 \
    --carryover_threshold 0.001 \
    --savename wf_ \
    --carryover_adetfile expanded_alpha.txt
```
This diagonalizes in the current subspace, saves the wavefunction, and writes the S+D-expanded determinants.

**Step 2 — Compute variance on expanded space:**
```bash
mpirun -np $NP ./diag \
    --fcidump fcidump.txt \
    --adetfile expanded_alpha.txt \
    --loadname wf_ \
    --iteration 0
```
This loads the wavefunction into the expanded space (zero-padded for new dets) and computes variance without diagonalizing.

**Next iteration:** Use `expanded_alpha.txt` as `--adetfile` and `wf_` as `--loadname` for the next Step 1. Loading the previous wavefunction provides a good initial guess for Davidson in the expanded space (zero-padded for new dets), which is much faster than restarting from the Hartree-Fock state.

### Tuning the Threshold

The `--carryover_threshold` parameter controls the growth rate of the subspace:

- **Larger threshold** (e.g., 0.01): fewer dets expanded, smaller growth per step, but each step is fast.
- **Smaller threshold** (e.g., 0.0001): more dets expanded, faster convergence, but each step is more expensive.
- **0.0**: expand from every det with nonzero amplitude.

A practical strategy is to start with a moderate threshold and tighten it as the subspace stabilizes:

```bash
# Iteration 1: aggressive filtering, fast exploration
--carryover_type 4 --carryover_threshold 0.01

# Iteration 2-3: moderate filtering
--carryover_type 4 --carryover_threshold 0.001

# Final iteration: include all contributing dets
--carryover_type 4 --carryover_threshold 0.0001
```

### Example: 29-Orbital System (995 Initial Determinants)

Starting from 995 sampled determinants (29 orbitals, 5 electrons per spin) with `--carryover_type 4 --carryover_threshold 0.001`:

| Step | N_dets | Energy (Ha) | Variance (Ha^2) | Expanded |
|------|--------|-------------|-----------------|----------|
| 0 | 995 | -101.9406 | 1.649 | 2881 |
| 1 | 2881 | -103.5749 | 0.0850 | 2881 |
| 2 | 2881 | -103.5913 | ~0 (1.6e-09) | 2881 |

- **Step 0**: The initial 995 sampled dets give energy -101.94 Ha. The large variance (1.65 Ha^2) indicates significant Hamiltonian-connected components outside the subspace.
- **Step 1**: After S+D expansion to 2881 dets and diagonalization (loading step 0's wavefunction as initial guess), the energy improves by 1.63 Ha and variance drops 20x to 0.085 Ha^2.
- **Step 2**: Rediagonalization in the same 2881-det space (loading step 1's wavefunction) gives an additional 0.016 Ha improvement with variance dropping to ~0, indicating the subspace is closed under the dominant Hamiltonian connections.

### Convergence Script

The script `scripts/variance_convergence.sh` automates this sequence:
```bash
# Basic usage
bash scripts/variance_convergence.sh ./diag fcidump.txt alpha_dets.txt

# With custom parameters
THRESHOLD=0.001 CARRYOVER_TYPE=4 MAX_DETS=50000 NP=4 \
    bash scripts/variance_convergence.sh ./diag fcidump.txt alpha_dets.txt
```

Environment variables:
- `NP` — MPI ranks (default: 1)
- `THRESHOLD` — amplitude cutoff for S+D extension (default: 0.001)
- `CARRYOVER_TYPE` — 4, 5, or 6 (default: 4)
- `MAX_DETS` — stop if expanded dets exceed this limit (default: 50000)
- `MAX_STEPS` — maximum number of iterations (default: 6)
- `MPIRUN_EXTRA` — additional mpirun flags (e.g., `--allow-run-as-root`)
