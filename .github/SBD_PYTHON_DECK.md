# SBD Python Walkthrough — slide deck draft

Audience: chemistry application experts who already use Dice as the
eigensolver inside `qiskit-addon-sqd`, evaluating whether to adopt SBD.
Co-presenter: domain expert handles chemistry interpretation; primary
presenter handles architecture / systems / perf / deployment.

Deck structure (6 slides + backup):

1. **What is SBD, and why** — positioned against Dice
2. **Tight integration via pybind11** — the architectural shift
3. **Hardware story** — backends and how to choose
4. **Roadmap (experimental)** — variance, subspace expansion, carryover variants *(co-presenter slide)*
5. **The HPC pain** — build-matrix and network-fabric differences across clusters
6. **The sqd-onboard agent** — the deployment helper
7. **Backup** — install commands, perf numbers, FAQ

Narrative arc: slides 1-4 establish that SBD is a faster, GPU-aware
alternative to Dice. Slide 5 names the cost of that speedup honestly
(every new cluster needs custom toolchain + MPI + fabric tuning).
Slide 6 is the answer: the agent does that work for you. Backup holds
the gritty install commands so the main flow stays clean.

Markers used in this draft:
- `[VERIFY]` — chemistry-interpretation claim that should be redlined by the co-presenter before delivery
- `[DIAGRAM]` — placeholder for a figure; description in speaker notes
- `[ASK]` — open question for the user / co-presenter

---

## Slide 1 — What is SBD, and why?

**Title:** SBD — GPU-accelerated alternative to Dice in SQD workflows
**Subtitle:** Same role inside `qiskit-addon-sqd`'s iteration loop; different process model and hardware envelope

### Body

[DIAGRAM] Two architecture diagrams, side by side:

```
qiskit-addon-sqd                      qiskit-addon-sqd  (MPI-aware fork)
       │                                       │
qiskit-addon-dice-solver                       │
       │                                       │
   subprocess(Dice CLI)              import sbd  ← in-process
       │                                       │
   *.bin output files                pybind11 → C++
       │                                       │
   _accelerate (Rust) parser         MPI + OpenMP + GPU
       │                                       │
   back to Python                    Davidson on the selected basis
```

### Bullets

- **SBD is the eigensolver.** Just like Dice. It diagonalizes the projected fermionic Hamiltonian on a user-provided list of α-determinants — it does NOT do the SQD outer loop (sample → recover → subsample → carryover). That outer loop is `qiskit-addon-sqd`'s job, same as today with Dice.
- **What changes when you switch to SBD:**
  - **In-process Python call** instead of subprocess + binary-file I/O per iteration.
  - **NVIDIA GPU + MPI accelerated** (CPU; NVIDIA Thrust via NVHPC `nvc++`; NVIDIA LLVM OMP-offload via `clang++`) — Dice is CPU+MPI only. AMD / Intel GPU not supported.
  - **Demonstrated to ~10⁹-dim subspaces** in production runs (Fe4S4 27,901² ≈ 778 M Cartesian basis on 8 GB200; orbital ceiling itself is the data-type bound in the comparison table).
  - **External selection model** — SBD takes the α-dets list as input and trusts it. Dice does its own internal SHCI heat-bath selection (the `eps` knob).
- **What stays the same:** FCIDUMP input format, 1-/2-RDM outputs, wavefunction save/load, carryover-determinant concept, role inside the SQD iteration loop.

### Comparison table

| | Dice (SHCI) | SBD |
|---|---|---|
| Hardware | CPU + MPI | CPU + MPI + **NVIDIA GPU** (Thrust / LLVM OMP-offload) |
| Process model | Subprocess + CLI + binary file I/O | In-process pybind11 module |
| Determinant selection | Internal (SHCI heat-bath, `eps` knob) | External (caller-provided list) |
| Orbital ceiling (data-type) | **128** (16-byte determinant address, hard-coded) ¹ | **160** at default `bit_length=20`; **512** at `bit_length=64` ² |
| 1-/2-RDM | ✓ | ✓ |
| Wavefunction save/load | ✓ | ✓ |
| Iteration with `qiskit-addon-sqd` | Stock package | MPI-aware fork (`@patch-ferminon-sbd`) |

¹ Per `qiskit-addon-dice-solver` README: *"determinant addresses are
interpreted by the Dice command line application to be 16-byte
unsigned integers; therefore, only systems of 128 or fewer orbitals
are supported."*

² SBD's ceiling is `SBD_MAX_DETSIZE × bit_length / 2`. The compile-
time `SBD_MAX_DETSIZE = 16` (in
[`vendor/sbd-upstream/include/sbd/chemistry/basic/omp_offload.h`](../vendor/sbd-upstream/include/sbd/chemistry/basic/omp_offload.h))
is the chunk-count cap; `bit_length` (default 20, runtime-configurable)
is bits-per-chunk. Raise `bit_length` for larger systems.

### Where it lives

| Layer | Repo |
|---|---|
| Upstream SBD (C++ core, RIKEN CCS) | https://github.com/r-ccs-cms/sbd |
| Python wrapper (fork with `python/` bindings + tooling) | https://github.com/hfwen0502/sbd |
| Patched `qiskit-addon-sqd` (MPI-aware solver hook) | https://github.com/hfwen0502/qiskit-addon-sqd (branch: `patch-ferminon-sbd`) |

### Speaker notes

- Open by recognizing the audience knows Dice. "If you already do SQD with Dice, this slide is the one-glance answer to 'what changes if I switch'."
- The architecture diagram does most of the work. Spend ~30 seconds on the in-process vs subprocess shift before moving to the table.
- The "external selection" point is worth pausing on: with Dice, the eigensolver decides which determinants to keep. With SBD, the caller (i.e., `qiskit-addon-sqd`'s recovery + subsample machinery) hands the list over. This means the noise-handling / configuration-recovery logic stays in Python where it's easy to inspect and tune. [VERIFY this framing with co-presenter — it's true mechanically but the chemistry-implications need their voice.]
- Have the perf numbers ready as a follow-up if anyone asks: 8×GB200 SBD Thrust 329 s vs Dice (their workload-specific number) — but don't volunteer perf on slide 1; that's slide 3.

### See also (in this repo)

- Example: standalone SBD diagonalization →
  [`python/examples/run_sbd_diag.py`](../python/examples/run_sbd_diag.py)
- Example: full SQD loop with SBD →
  [`python/examples/run_sqd_sbd.py`](../python/examples/run_sqd_sbd.py)
- Examples README (h2o, n2, fe4s4 walkthroughs) →
  [`python/examples/README.md`](../python/examples/README.md)
- Bundled inputs: H2O ([fcidump](../data/h2o/fcidump.txt) +
  [α-dets 1e-3](../data/h2o/h2o-1em3-alpha.txt)), Fe4S4
  ([fcidump](../data/fe4s4/fcidump.txt) +
  [α-dets 27,901](../data/fe4s4/alpha_dets_27901.txt))

### Open questions

- *(resolved)* Orbital-ceiling claim corrected: Dice = 128 (16-byte determinant address, per `qiskit-addon-dice-solver` README's *Limitations* section); SBD = `SBD_MAX_DETSIZE × bit_length / 2` (160 at default, 512 at `bit_length=64`).

---

## Slide 2 — Tight integration via pybind11

**Title:** Tight integration via pybind11 — SBD is a Python function, not a subprocess
**Subtitle:** What changes for `qiskit-addon-sqd` users

### Body

[DIAGRAM] Two stacked timelines, one per SQD iteration:

```
Dice path (per iteration):
   Python → write FCIDUMP to disk → fork(Dice CLI) →
   wait → parse *.bin output (Rust _accelerate) →
   back to Python with energy/RDMs

SBD path (per iteration):
   Python → run_sbd_diag(...) → C++ Davidson runs in same
   process, on this MPI rank, on this GPU →
   returns Python objects directly
```

### How you launch the job

The launch command itself reflects the architectural difference:

| Solver | Launch command | Why |
|---|---|---|
| Dice | `python my_script.py`                | The Python script runs serially. Each `solve_fermion(...)` call internally forks `mpirun + bin/Dice` for that one diagonalization. |
| SBD  | `mpirun -np N python my_script.py`   | SBD is MPI-native — you launch N Python ranks once, each `import sbd` loads the C++ kernel into that rank, and the SQD outer loop stays in-process across iterations. |

### Bullets

- **One Python module, three backends.** `import sbd` loads `_core_cpu.so`, `_core_gpu.so`, and/or `_core_gpu_omp_nvidia.so` — whichever wheels match what's installed. Backend chosen at call time via `device='cpu' | 'gpu' | 'gpu-omp'`. See [`python/__init__.py`](../python/__init__.py) and [`python/device_config.py`](../python/device_config.py).
- **The pybind11 surface** ([`python/bindings.cpp`](../python/bindings.cpp), ~400 LOC) directly exposes the C++ Davidson + the FCIDUMP loader + the α-dets loader as Python callables. No CLI. No file-format parser layer.
- **What this enables, concretely:**
  - **No FCIDUMP-to-disk + bin-file-parse round-trip per SQD iteration.** Inputs are ndarrays; outputs are ndarrays.
  - **MPI-aware:** each rank does `import sbd` once at startup and stays in-process across iterations. No fork/exec storm at high node counts.
  - **Python-debuggable:** stack traces cross the C++/Python boundary. Set a `breakpoint()` before the SBD call, inspect intermediate state, return to Python with the SCIResult.
  - **Composable:** drop SBD inside any Python loop — custom convergence checks, custom carryover policies, on-the-fly RDM post-processing — without writing more C++.
- **`qiskit-addon-sqd` integration:** `solve_fermion`-shaped wrapper at
  [`python/sbd_solver.py`](../python/sbd_solver.py) is a near-drop-in
  for `qiskit_addon_dice_solver.solve_fermion`. It takes ci_strings +
  `one_body_tensor` + `two_body_tensor` + nelec / norb and returns an
  `SCIResult` — same shape Dice's wrapper returns. Requires the
  MPI-aware fork: `pip install git+https://github.com/hfwen0502/qiskit-addon-sqd@patch-ferminon-sbd`.

### Code: see what your install has

`import sbd` eagerly loads every backend that was compiled, so
runtime introspection is a one-liner:

```python
import sbd
sbd.available_backends()
# ['cpu', 'gpu', 'gpu-nvidia-omp']    ← whichever .so files are built

sbd.print_info()
# ============================================================
# SBD (Selected Basis Diagonalization) Python Bindings
# ============================================================
# Version: 1.5.0
# Compiled backends: cpu, gpu, gpu-nvidia-omp
# ...
```

Same idea from the shell:

```bash
python -c "import sbd; print(sbd.available_backends())"
```

### Code: switching backends — three equivalent paths

Same call, three places to pick the device:

```python
# 1. Set a process-wide default at startup
import sbd
sbd.init(device='gpu')                   # all subsequent calls use gpu

# 2. Override per-call without re-initializing
result_cpu = sbd.tpb_diag(..., device='cpu')      # debugging
result_gpu = sbd.tpb_diag(..., device='gpu')      # production
result_omp = sbd.tpb_diag(..., device='gpu-omp')  # alternative kernel
# All three return bit-equal energies.

# 3. Or pass it through DeviceConfig (the qiskit-addon-sqd path)
from sbd.device_config import DeviceConfig
energy, sci_state = solve_sci(..., device_config=DeviceConfig.gpu())
```

And from the command line — every shipped example script accepts
`--device`:

```bash
python run_sbd_diag.py --device cpu       # OpenMP-on-CPU path
python run_sbd_diag.py --device gpu       # NVHPC Thrust path
python run_sbd_diag.py --device gpu-omp   # LLVM offload path
```

→ Source: [`python/__init__.py`](../python/__init__.py)
(`available_backends`, `get_backend`, `init`, `print_info`),
[`python/device_config.py`](../python/device_config.py) (`DeviceConfig`,
`get_device_info`).

### Code: same call shape, very different invocation

The user-facing API looks the same:

```python
# Dice
from qiskit_addon_dice_solver import solve_fermion
energy, sci_state = solve_fermion(bitstring_matrix, hcore, eri, spin_sq=0.0)

# SBD
from sbd.sbd_solver import solve_sci
from sbd.device_config import DeviceConfig
energy, sci_state = solve_sci(ci_strings, one_body_tensor, two_body_tensor,
                              norb=norb, nelec=nelec,
                              device_config=DeviceConfig.gpu(),
                              mpi_comm=MPI.COMM_WORLD)
```

…but what the wrappers do internally is the real story.

**Inside Dice's wrapper** ([`qiskit_addon_dice_solver/dice_solver.py:_call_dice`](https://github.com/Qiskit/qiskit-addon-dice-solver/blob/main/qiskit_addon_dice_solver/dice_solver.py)):

```python
# Per call: make a temp dir, write FCIDUMP + input.dat to disk,
# fork mpirun + Dice CLI, parse binary output files back from disk.
dice_dir   = Path(tempfile.mkdtemp(prefix="dice_cli_files_", ...))
tools.fcidump.from_integrals(dice_dir / "fcidump.txt", hcore, eri, norb, nelec)
_write_input_files(ci_strs=ci_strs, ..., dice_dir=dice_dir)

dice_call  = ["mpirun", *mpirun_options, "<…>/qiskit_addon_dice_solver/bin/Dice"]
with open(dice_log_path, "w") as logfile:
    subprocess.run(dice_call, cwd=dice_dir, stdout=logfile, stderr=logfile)

e_dice, sci_state, occupancies = _read_dice_outputs(dice_dir, norb, nelec, ...)
shutil.rmtree(dice_dir)   # clean up the temp dir
```

**Inside SBD's wrapper** ([`python/__init__.py`](../python/__init__.py),
[`python/sbd_solver.py`](../python/sbd_solver.py)):

```python
# In-process: route to the per-device pybind11 module and call C++ directly.
def tpb_diag(fcidump, adet, bdet, sbd_data, ..., device=None):
    backend = get_backend(device)              # _core_cpu / _core_gpu / _core_gpu_omp_nvidia
    return backend.tpb_diag(fcidump, adet, bdet, sbd_data, ...)
```

What this means at the SQD scale: **5 SQD iterations × N MPI ranks** =
5×N fork/exec cycles + 5 FCIDUMP-write + 5 binary-file-parse round-trips
on the Dice path, all of which collapse to a single `import sbd` and N
in-process function calls on the SBD path.

It's also what makes the runtime backend switch above (`device='cpu'`
→ `device='gpu'` → `device='gpu-omp'`) free on SBD. Dice has no
analogous lightweight switch — every call is its own subprocess
regardless of what you change.

### Code: full SQD loop with SBD, taking bitstrings as input

This is the real ~30-line spine of
[`python/examples/run_sqd_sbd.py`](../python/examples/run_sqd_sbd.py),
distilled. **The user's quantum-measurement bitstrings come in as a
`BitArray`; SBD comes in as the `sci_solver=` callable. Everything
in between is `qiskit-addon-sqd`'s own machinery (sample → configuration
recovery → subsample into batches → diagonalize → carryover → repeat).**

```python
import json, numpy as np
from functools import partial
from mpi4py import MPI
from pyscf import ao2mo, tools
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

import sbd
from sbd.sbd_solver import solve_sci_batch
from sbd.device_config import DeviceConfig

# 1. Load Hamiltonian from FCIDUMP
mf    = tools.fcidump.to_scf("data/h2o/fcidump.txt")
hcore = mf.get_hcore()
eri   = ao2mo.restore(1, mf._eri, norb)

# 2. Bitstrings — from quantum-measurement counts. count_dict_*.json
#    files are bundled in python/examples/ for h2o, n2, fe4s4.
counts  = json.load(open("python/examples/count_dict_h2o.json"))
joined  = "".join(counts.keys())
matrix  = (np.frombuffer(joined.encode(), dtype=np.uint8) == ord("1")).reshape(
              len(counts), -1)
matrix  = np.repeat(matrix, list(counts.values()), axis=0)
bit_array = BitArray.from_bool_array(matrix)

# 3. Wire SBD into qiskit-addon-sqd's `sci_solver=` slot
sbd.init(device='gpu')
sbd_solver = partial(
    solve_sci_batch,
    mpi_comm    = MPI.COMM_WORLD,
    sbd_config  = {"method": 0, "max_it": 100, "max_nb": 50,
                   "carryover_type": 1, "ratio": 0.1, "threshold": 1e-4},
    device_config = DeviceConfig.gpu(),
    fcidump_path  = "data/h2o/fcidump.txt",
)

# 4. Run the self-consistent SQD loop
result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    norb              = norb,
    nelec             = (num_elec_a, num_elec_b),
    samples_per_batch = 300,
    num_batches       = 3,
    max_iterations    = 5,
    sci_solver        = sbd_solver,    # ← SBD plugs in here
)
```

Same shape as plugging in `qiskit_addon_dice_solver.solve_fermion` —
only the `sci_solver=` callable changes. The SQD outer loop, the
configuration-recovery step, and the bitstring-to-determinant
conversion all stay in `qiskit-addon-sqd`.

→ Full driver: [`python/examples/run_sqd_sbd.py`](../python/examples/run_sqd_sbd.py)
→ Bundled bitstring inputs:
  [`count_dict_h2o.json`](../python/examples/count_dict_h2o.json) (24 orbitals),
  [`count_dict_n2.json`](../python/examples/count_dict_n2.json) (60 orbitals),
  [`count_dict_fe4s4.json`](../python/examples/count_dict_fe4s4.json) (36 orbitals)

### Install

Build is non-trivial across clusters (different toolchain per backend,
different MPI vendor per box, different network fabric flags). Slides
5–6 cover that pain and how the sqd-onboard agent handles it. Exact
pip commands are in the **Backup** slide.

### Speaker notes

- This slide is intentionally code-heavy and will likely **split into
  two PowerPoint slides**: (a) the diagram + bullets + "what's
  available" + "switching backends" snippets; (b) the Dice→SBD swap +
  the full SQD-loop snippet. Pacing: 60 sec on the architecture
  framing, 30 sec on the introspection/switching block, 90 sec on
  the SQD-with-bitstrings walkthrough.
- For the SQD-with-bitstrings code: read it as 4 numbered steps, not
  line-by-line. The audience needs to recognize their workflow in it,
  not memorize the API. The point is **"your quantum-measurement
  bitstrings go in here, the energy comes out, SBD is just the
  `sci_solver=` callable in the middle."**
- The MPI-awareness point is worth pausing on: with Dice, each
  iteration forks a fresh CLI process; at 8-rank+ that fork/exec
  storm becomes meaningful overhead. With SBD, ranks stay alive
  across the full SQD loop.
- Mention the debugging benefit briefly — Python users notice this
  when they hit a bug. Stack traces from inside Davidson land in the
  user's Python session, not in a `dice.out.X` file.
- The "drop-in" framing is API-level: same call shape, same return
  type. The user still has to install the MPI-aware fork of
  `qiskit-addon-sqd` and pick a backend at call time.
- The two-pass build line (deferred to backup) is a real footgun if
  anyone asks "how do I build it" — call out that the agent (slide 6)
  handles this for them.

### See also (in this repo)

- pybind11 surface → [`python/bindings.cpp`](../python/bindings.cpp)
- qiskit-addon-sqd-compatible wrapper → [`python/sbd_solver.py`](../python/sbd_solver.py)
- Backend dispatch + device config → [`python/__init__.py`](../python/__init__.py),
  [`python/device_config.py`](../python/device_config.py)
- Standalone example → [`python/examples/run_sbd_diag.py`](../python/examples/run_sbd_diag.py)
- SQD-with-SBD end-to-end example →
  [`python/examples/run_sqd_sbd.py`](../python/examples/run_sqd_sbd.py)
- Sample bitstring inputs (count_dict format) →
  [`python/examples/count_dict_h2o.json`](../python/examples/count_dict_h2o.json),
  [`count_dict_n2.json`](../python/examples/count_dict_n2.json),
  [`count_dict_fe4s4.json`](../python/examples/count_dict_fe4s4.json)
- MPI-aware `qiskit-addon-sqd` fork →
  https://github.com/hfwen0502/qiskit-addon-sqd/tree/patch-ferminon-sbd

### Open questions

- [ASK] Do you want me to verify the exact `solve_sci` signature against the current `python/sbd_solver.py` source before delivery? The snippet above is paraphrased — minor parameter names may have drifted. (I can pull the real signature for the slide.)

---

## Slide 3 — Hardware story: three backends, one decision

**Title:** Three backends, one runtime decision: where does this run?
**Subtitle:** Same source, same energies, different compilers — pick by what's installed

### Body — backend characteristics

| Backend (`device=`) | Compiler | When to use it |
|---|---|---|
| `cpu`                       | system `c++` (gcc/clang) | small problems; debugging; no GPU available |
| `gpu` (Thrust)              | NVHPC `nvc++`            | NVIDIA GPU; production default |
| `gpu-omp` (LLVM offload)    | LLVM `clang++` w/ NVPTX  | NVIDIA GPU; alternative kernel path |

All three are loaded at `import sbd` if their `.so` was built. Switch
at call time:

```python
energy, sci_state = solve_sci(..., device_config=DeviceConfig(device='gpu'))
# or 'gpu-omp', or 'cpu' — same call shape, no rebuild
```

### Body — what each backend earns you

For `--iteration 1 --block 10` (10 Davidson sub-iters) on Fe4S4 27,901
α-determinants on coreweave GB200, post-rebuild with native sm_100:

| Backend | 4×GB200 (1 node) | 8×GB200 (2 nodes) | 1n→2n |
|---|---:|---:|---:|
| Thrust       | **577 s** (Davidson 555 s · setup 33 s · final 55 s)  | **329 s** (Davidson 320 s) | 1.75× |
| OMP-offload  | **491 s** (Davidson 437 s · setup 38 s · final 35 s)  | **304 s** (Davidson 267 s) | 1.62× |

Energies are **bit-equal across cpu, gpu (Thrust), gpu-omp, and 1n/2n
configurations** at −326.821832430028. Backend choice doesn't move
the eigenvalue — it moves the wallclock.

**Reference for the audience**: this is the same workload Dice runs
on CPU + MPI inside `qiskit-addon-sqd` today. We haven't measured Dice
on this exact configuration, so leave the speedup framing to the
audience's own CPU baseline — what's solid is that the chemistry
result (energy, RDMs) is unchanged, and the absolute wallclocks above
are measured.

### How to choose

Most sites will land on one default and stick. A pragmatic rule:

- **Has a recent NVHPC SDK** (most NVIDIA-shop clusters) → start with
  `gpu` (Thrust). The build is the simplest; perf is competitive.
- **Doesn't have NVHPC, has LLVM with NVPTX target** → `gpu-omp`.
  Same hardware, different kernel implementation. On Blackwell
  today, `gpu-omp` is *slightly* faster than Thrust at 2-node (304 s
  vs 329 s); on Hopper it tracks Thrust closely.
- **CPU only or debugging** → `cpu`. Fast for h2o-class problems;
  not for production fe4s4-scale.

The `gpu`/`gpu-omp` choice is hardware-and-toolchain dependent, not a
chemistry decision. Pick whichever your cluster ships with cleanly.

### Speaker notes

- This slide is the headline perf moment. Lead with the table, not
  the decision tree.
- We do NOT have a measured Dice number on this workload. Resist the
  urge to quote a speedup-vs-Dice multiple from memory — the audience
  knows their own CPU baseline and will fill that in. Stick to:
  *"the absolute wallclocks above are measured; energies are
  bit-equal across backends."* If asked, say so honestly.
- The bit-equal-across-backends point is important: it's a
  correctness claim that earns the audience's trust before slide 4
  (where we ask them to consider new experimental features).
- If anyone asks "why does gpu-omp edge out Thrust on 2-node?" — the
  per-Davidson-sub-iter timing (27 s vs 32 s) is a real but small
  effect; on a different workload it could flip. Don't sell either
  as the universal winner.

### See also (in this repo)

- Backend dispatch + DeviceConfig →
  [`python/__init__.py`](../python/__init__.py),
  [`python/device_config.py`](../python/device_config.py)
- Runtime device selection in the standalone example →
  [`python/examples/run_sbd_diag.py`](../python/examples/run_sbd_diag.py) (the `--device` flag)
- Full perf breakdown including per-matvec exch/compute split for
  Fulqrum (which DOES expose communication primitives directly) →
  [`FULQRUM_SBD_GB200_SCALING.md`](./FULQRUM_SBD_GB200_SCALING.md) §1
  *"Combined view"*

### Open questions

- [ASK] If a measured Dice wallclock on Fe4S4 27,901 (10 Davidson sub-iters or equivalent) becomes available — even a single CPU number from a known box — we can add a Dice row to the perf table and have a real comparison. Until then, no speedup multiples are quoted; the slide stays at "energies bit-equal, absolute SBD wallclocks measured."

---

## Slide 4 — Roadmap (main vs experimental branch)

**Title:** Roadmap — `main` vs `singles-doubles-extend`
**Subtitle:** *Co-presenter slide* — chemistry interpretation by [domain expert]

### Where stable vs experimental code lives

| Branch | SBD C++ source | What it has |
|---|---|---|
| `main` (stable)          | **submodule** → upstream `r-ccs-cms/sbd` (unmodified)  | Python wrapper, three backends, SQD integration |
| `singles-doubles-extend` | **embedded in fork** (C++ modified for new features)   | + variance, S+D expansion, ERI screening, TrimSQD |

Both at https://github.com/hfwen0502/sbd · `main` vs `tree/singles-doubles-extend`.
Experimental features documented in
[`apps/chemistry_tpb_selected_basis_diagonalization/VARIANCE.md`](https://github.com/hfwen0502/sbd/blob/singles-doubles-extend/apps/chemistry_tpb_selected_basis_diagonalization/VARIANCE.md).

Why the structural difference matters for adoption: on `main` the
upstream SBD pin is a submodule, so SBD bug fixes flow in cleanly via
`git submodule update`. On `singles-doubles-extend` the C++ is
fork-owned because the new carryover types and `--iteration 0` mode
need direct edits to SBD's C++ source.

### Body

Three new capabilities on the experimental branch, exposed via new
`--carryover_type` values and an `--iteration 0` mode.

#### 1. Singles + Doubles subspace expansion (`--carryover_type 4-8`)

Expand the SBD-selected basis with single + double excitations from
selected determinants. Existing types 1–3 already extend with
**singles only**; new types 4–6 add **same-spin doubles** on top.

| Type | Selection           | Extension            |
|------|---------------------|----------------------|
| 4    | Amplitude           | Singles + Doubles    |
| 5    | Marginal+amplitude  | Singles + Doubles    |
| 6    | None (all dets)     | Singles + Doubles    |

For an n-occupied / m-virtual half-determinant, brute-force S+D adds
n·m + C(n,2)·C(m,2) excitations — quickly explosive.

#### 2. ERI-screened S+D (`--carryover_type 7-8`)

Same S+D expansion, but **filter by Hamiltonian integral magnitude**:
keep an excitation only if its Fock-element (singles) or
antisymmetrized 2e-integral (doubles) exceeds `--eri_threshold`.

| Type | Selection | Extension       |
|------|-----------|-----------------|
| 7    | Amplitude | Screened S + D  |
| 8    | None      | Screened S + D  |

Typically keeps 20–50% of brute-force S+D excitations while retaining
the physically important ones (those with non-negligible Hamiltonian
coupling).

#### 3. Variance-only mode (`--iteration 0`) and the extrapolation workflow

Skip diagonalization; load a pre-computed wavefunction (`--loadname`),
compute one matvec `H|ψ⟩`, and report:
- **Energy** ⟨ψ|H|ψ⟩ / ‖ψ‖²
- **Variance** σ² = ⟨Hψ|Hψ⟩ / ‖ψ‖² − E²

Pair this with S+D expansion in a two-step protocol:

```
[diag in S]  →  save wf       →  [variance in S']  →  repeat with S' as new S
   ↑                                    ↓
expanded to S' via --carryover_type 4/7
```

Iterating until σ² → 0 extrapolates to the exact eigenvalue. The
zero-variance-limit pair (E, σ²) drives the convergence diagnostic
and (optionally) a Richardson-style energy extrapolation.

#### 4. TrimSQD — adaptive subspace pruning

Between expansion rounds, **trim** the determinant set: rediagonalize
in the expanded space, then keep only dets with marginal amplitude
above `TRIM_THRESHOLD`.

Concrete demo from `VARIANCE.md` — `NORB=29, nelec=(5α, 5β)`
(10 electrons total, MS2=0), seeded from 995 sampled determinants:

| Step | dets (no trim) | dets (TrimSQD) | Energy (Ha) | Variance (Ha²) |
|------|---------------:|---------------:|------------:|---------------:|
| 0    | 995            | 995            | −101.9406   | 1.649          |
| 1    | 879            | 656            | −103.18     | 0.530          |
| 2    | 11,289         | 4,442          | −103.59     | 0.007          |
| 3    | 11,042         | **5,794**      | **−103.5938** | 0.001        |

Both reach the same energy (0.16 mHa from FCI). TrimSQD does it with
**47% fewer dets** in the final subspace — payoff scales with system
size.

### Speaker notes

- Defer chemistry-impact questions to the co-presenter. The
  primary-presenter's territory: the `--carryover_type` matrix, the
  CLI knobs, the workflow diagram, the demo table. Co-presenter takes
  the *"what does variance mean physically; how does ERI screening
  preserve correlation; how does TrimSQD compare to SONIC"* questions.
- The 29-orbital convergence example is the punchline of the slide.
  If pressed for time, drop everything else and keep the 4-row demo
  table — it lands the *"this works"* claim without needing
  interpretation.

### Open questions

- [ASK] Co-presenter to redline the chemistry-impact framing —
  particularly the "variance → exact eigenvalue extrapolation"
  characterization.
- [ASK] Is the `qiskit-addon-sqd` outer loop already wired to use
  `--carryover_type 4-8` from Python? If yes, link the example
  driver. If no, this is a CLI-only feature for now and worth being
  explicit about.

### See also

- Branch: [`singles-doubles-extend`](https://github.com/hfwen0502/sbd/tree/singles-doubles-extend)
- Variance test driver: [`python/examples/test_variance.py`](../python/examples/test_variance.py)
- Per-iteration variance use in the SQD loop (if available) — the
  [`python/examples/run_sqd_sbd.py`](../python/examples/run_sqd_sbd.py)
  driver is the natural place to demonstrate it. [VERIFY whether
  current driver wires up the variance call.]

---

## Slide 5 — The HPC pain: build matrix and network fabrics

**Title:** Why standing this up on a new cluster is harder than `pip install`
**Subtitle:** Toolchain × MPI vendor × network fabric — every box is different

### Body — the build matrix

SBD has three backends. Each needs a different compiler.
distutils only supports ONE `CXX` per `setup()` call, so the install
takes **two pip invocations** to get everything:

| Backend | Compiler | Use case |
|---|---|---|
| `_core_cpu`            | system `c++` (gcc / clang) | debugging, small problems, no GPU |
| `_core_gpu` (Thrust)   | NVHPC `nvc++` (CUDA + Thrust) | production GPU on NVIDIA |
| `_core_gpu_omp_nvidia` | LLVM `clang++` w/ NVPTX target | alternative GPU path |

Plus a long tail of environment plumbing per box:

- **GPU compute capability** must match hardware exactly. Default is
  `sm_90` (Hopper); on Blackwell you need `sm_100`. Get this wrong and
  the LLVM offload path silently host-falls-back with uninitialized
  memory — your eigenvalue is wrong, your wallclock is meaningless,
  and nothing errors. (We hit this. Cost: half a day.)
- **MPI vendor** dictates launch flags. HPCX 4.1.x cannot init
  `pml=ucx` inside a SLURM cgroup; you fall back to
  `ob1 + smcuda + tcp`. Stock OpenMPI 5.x is fine. Spectrum MPI on LSF
  needs `jsrun` and different binding semantics.
- **mpi4py ABI** must match the runtime MPI. Pip-installed wheels
  built against OpenMPI 5 segfault on HPCX 4. Source rebuild required.
- **`LD_LIBRARY_PATH` ordering** for `libomp.so` — LLVM's lib must
  precede NVHPC's, otherwise the wrong OpenMP runtime loads and
  offload misbehaves.

### Body — the network-fabric matrix

Same SBD source code. Four representative clusters:

| Cluster | Hardware | Fabric | Best comm backend | Why |
|---|---|---|---|---|
| coreweave GB200 (2 nodes) | 8× GB200, NVL4 + IB | **MNNVL** across nodes | `nccl` via P2P/MNNVL | NCCL routes cross-node over NVLink, not IB → 4.8× exch speedup over `cuda_mpi` |
| 8× H100 (1 node)          | NVLink + NVSwitch + PCIe Gen5 | intra-node only | `nccl` or `cuda_mpi` (similar) | No MNNVL, no GPUDirect peermem on some boxes; comm dwarfed by compute |
| IBM LSF / jsrun cluster   | Spectrum MPI + IB | IB + GDR | `cuda_mpi` (Spectrum is CUDA-aware) | jsrun launcher, IBM's MPI ABI; different launch / binding from SLURM |
| Plain VM (no IB)          | single node, TCP only | none cross-node | `host_mpi` or `nccl` intra-node | No high-speed fabric; cross-node not viable |

The same Fulqrum + SBD code reaches **188 s on 2-node GB200 with NCCL+MNNVL** but **264 s with cuda_mpi over TCP-staged IB** — the *fabric* choice, not the kernel, dominates.

### What goes wrong if any of this is mis-set

| Symptom | Root cause | Cost to find |
|---|---|---|
| Davidson reports `tol=0` at iter 0; energy looks like HF | gpu-omp built `sm_90`, hardware is `sm_100` → silent host fallback | hours |
| `mpirun` prints `Executable: \` and dies | MPIRUN_OPTS heredoc preserves `\<NL>` literally inside `"..."` | half an hour |
| All ranks pile onto GPU 0; other GPUs idle | wrapper used `SLURM_LOCALID` (alloc-wide) instead of `OMPI_MCA_orte_ess_node_rank` (per-task) | an hour |
| `Requested node configuration is not available` | SLURM `DefMemPerCPU × cpus-per-task > RealMemory`; need explicit `--mem` | confusing, opaque |
| `PML ucx cannot be selected` | HPCX 4.1.x can't init UCX inside SLURM cgroup | obvious once you know |

These are real findings from one cluster's onboarding. A new cluster
will surface a fresh set.

### Speaker notes

- This slide is the bridge. Don't dwell on individual rows. Spend
  ~30 sec on the build-matrix table, ~30 sec on the fabric table, ~30
  sec on the failure-mode table. The audience reaction you want is
  *"so I'd have to figure all of that out per cluster"* — yes,
  exactly, and the next slide is the answer.
- The MNNVL → 4.8× exch number is a strong hook because it's a
  cluster-architecture decision the chemist doesn't control, and it
  changes which comm backend they should run.
- The "fresh set per cluster" line is honest and important — the
  agent doesn't know everything; it discovers per-cluster signatures
  and grows the playbook. That sets up slide 6's framing of
  "knowledge captured in `playbook/signatures.yaml`."

### See also

- Today's per-cell perf numbers, including the matvec breakdown that
  shows the fabric impact directly →
  [`FULQRUM_SBD_GB200_SCALING.md`](./FULQRUM_SBD_GB200_SCALING.md)

---

## Slide 6 — The sqd-onboard agent

**Title:** Letting an agent absorb the HPC stack
**Subtitle:** UCX, NCCL, MPI, fabric, launcher — discovered per cluster, not encoded in your head

### Body — the framing

Single-node SBD is mostly tractable: install NVHPC or LLVM, `pip install`,
run `mpirun -np 4`. Cluster-vendor HPC stacks bite at **2+ nodes**, where
choices compound:

- **UCX** transport selection (`cuda_copy` / `cuda_ipc` / `rc` / `ud`)
  and which transports the cluster actually exposes inside its cgroup.
- **NCCL** version (≥ 2.23 needed for MNNVL) and which fabric it
  picks (P2P over MNNVL vs IB+GDR vs sockets).
- **MPI vendor** (HPCX vs OpenMPI 5 vs Spectrum) and the launcher
  (`mpirun` vs `srun --mpi=pmix` vs `jsrun`), each with different
  CUDA-aware semantics and binding behavior.
- **GPU-pinning convention** (which env var the wrapper reads to pick
  per-rank `CUDA_VISIBLE_DEVICES`) — different MPI implementations
  expose different vars before MPI_Init.
- **Fabric peculiarities** (MNNVL on GB200, GPUDirect peermem
  loaded/missing on H100, no high-speed fabric on a VM) — pick the
  wrong comm backend and you leave 1.4–4.8× on the table.

A chemistry user shouldn't have to know any of this to run an SQD job.

### What the agent does

The `sqd-onboard` agent (separate repo) is a deployment helper. It
walks the user's cluster, picks the right backend / launcher / fabric
flags, builds the stack, validates correctness against a small
reference (h2o), and emits ready-to-submit run scripts under
`run/<solver>/<n>node/`. No HPC tuning needed from the chemist.

When the agent encounters a failure mode it hasn't seen, the fix is
captured as a **signature** in a shared playbook so the next user on
that cluster — or a similar one — doesn't repeat the discovery.

### Speaker notes

- This slide is intentionally abstract. Don't tour features; just
  land the message: the inherited HPC stack is real and complex, and
  the agent absorbs it.
- The 2+ node framing is the punchline. Most chemists run single-node
  and don't feel this pain. The moment they want to scale to multi-
  node — which is where SBD's perf headroom lives (188 s vs 304 s
  going 1n→2n is the kind of number this slide implicitly justifies) —
  the stack complexity hits.
- If anyone asks "show me the agent," that's a backup-slide demo or a
  follow-up conversation — not a 30-second slide answer.

### See also

- The `sqd-onboard` repository (separate) → contains the agent prompt,
  the signature playbook capturing per-cluster failure modes, the
  build/run script templates referenced above.

---

## Slide 7 — Backup

### Install commands (full)

```bash
# CPU + Thrust GPU (NVHPC nvc++ on PATH)
SBD_BUILD_BACKEND=both CC=nvc CXX=nvc++ \
    pip install --no-build-isolation -e .

# OMP-offload GPU (LLVM clang++ with NVPTX target on PATH; built separately
# because it's incompatible with NVHPC's CXX in a single setup() call)
SBD_BUILD_BACKEND=gpu_omp_nvidia \
    pip install --no-build-isolation -e .

# qiskit-addon-sqd MPI-aware fork (required for SBD inside the SQD loop)
pip install git+https://github.com/hfwen0502/qiskit-addon-sqd@patch-ferminon-sbd
```

### Reference perf numbers

→ [`FULQRUM_SBD_GB200_SCALING.md`](./FULQRUM_SBD_GB200_SCALING.md) §1
"Combined view" has the wall-time + per-matvec breakdown for SBD
(Thrust, OMP-offload) and Fulqrum (nccl, cuda_mpi, host_mpi) at 1n
and 2n on coreweave GB200.

### FAQ

*(To be drafted closer to delivery — typical questions: how does
SBD handle different (na, nb) electron counts; what symmetry sectors
are supported; how is convergence reported; what's the carryover
mechanism's effect on iteration count, etc.)*

---

*All slides drafted. Iterate / verify / refine before PowerPoint export.*
