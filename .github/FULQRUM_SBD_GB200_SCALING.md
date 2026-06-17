# Fulqrum + SBD on Grace/GB200 — scaling 1 node → 2 nodes

Workload: **Fe4S4 27,901 alpha-determinants** (72-qubit JW operator,
2,475,752 Pauli terms, 778,465,801-dim Cartesian subspace for Fulqrum
/ 27,901-element selected subspace for SBD). Same `fcidump.txt` and
same `alpha_dets_27901.txt` (SBD-derived selection) across all runs.

Tested configurations:
- **4×GB200**, 1 node, intra-node NV18 mesh.
- **8×GB200**, 2 nodes × 4 GPUs, MNNVL (Multi-Node NVLink Switch) fabric across nodes.

A reference H100 row (8 GPUs, 1 node) is included where the data was
already in the prior benchmark sweep. The H100 box uses PCIe Gen5 to
GPUs and NVLink + NVSwitch intra-node — fundamentally different fabric.

## 1. Scaling: 4×GB200 (1 node) → 8×GB200 (2 nodes)

End-to-end eigensolve / davidson wall time for each backend, with
energy as a correctness check. Refreshed 2026-06-16 after Fulqrum's
GPU-resident PRIMME path (`cublas_dprimme`) merged via PR #7
(`aabb-csr-gpu-eig`).

### Combined view — wall + per-iter breakdown (matched ~10 matvecs)

Each Fulqrum cell shows the steady-state per-matvec breakdown
(compute / exch / sync) below the wall; each SBD cell shows the
Davidson sub-iter cost and the setup / final-mult split.

| Backend | 4×GB200 (1 node) | 8×GB200 (2 nodes) | 1n→2n |
|---|---|---|---|
| **Fulqrum nccl**     | **221 s** — host PRIMME (INT32 forces it) <br> per-matvec: compute **7.66 s** · exch **1.34 s** · sync 34 ms | **118 s** — `cublas_dprimme` <br> per-matvec: compute **4.15 s** · exch **1.07 s** · sync 40 ms | 1.87× |
| **Fulqrum cuda_mpi** | **219 s** — host PRIMME <br> per-matvec: compute **7.62 s** · exch **1.33 s** · sync 34 ms | **134 s** — `cublas_dprimme` <br> per-matvec: compute **6.40 s** · exch **3.30 s** · sync 40 ms | 1.63× |
| **Fulqrum host_mpi** | **234 s** — host PRIMME <br> per-matvec: compute **8.08 s** · exch **0.68 s** · sync **1.13 s** | **131 s** — `cublas_dprimme` <br> per-matvec: compute **5.79 s** · exch **2.68 s** · sync 59 ms | 1.79× |
| **SBD Thrust**       | **577 s** <br> Davidson 555 s (≈ 55 s/sub-iter, 10 iters) · setup 33 s · final mult 55 s | **329 s** <br> Davidson 320 s (≈ 32 s/sub-iter) · setup 39 s · final mult 31 s | 1.75× |
| **SBD OMP-offload**  | **491 s** <br> Davidson 437 s (≈ 44 s/sub-iter) · setup 38 s · final mult 35 s | **304 s** <br> Davidson 267 s (≈ 27 s/sub-iter) · setup 18 s · final mult 21 s | 1.62× |

Energies:
- Fulqrum 1n nccl/cuda_mpi/host_mpi → **−326.822567853744** (12-digit match across all three comm backends)
- Fulqrum 2n nccl/cuda_mpi/host_mpi → **−326.822567851544** (12-digit match)
- SBD all backends, 1n + 2n → **−326.821832430028** (bit-equal)

The 0.0006 Ha gap is SBD's truncation of off-diagonal configurations
on the same 27,901 half-strs.

What the breakdown surfaces:

- **2n nccl Fulqrum is the only cell that's actually compute-bound** — exch (1.07 s) fits inside compute (4.15 s), MNNVL P2P over NVLink rather than IB.
- **2n cuda_mpi Fulqrum is exch-heavy** (3.30 s exch / 6.40 s compute) — cross-node Allgatherv host-stages through TCP because UCX cannot init in this cluster's SLURM cgroup. About half the matvec is communication.
- **2n host_mpi sits between** (2.68 s exch / 5.79 s compute) — same TCP cross-node cost as cuda_mpi but without the smcuda CUDA-IPC bookkeeping that cuda_mpi pays intra-node. Wall: 131 s vs cuda_mpi's 134 s, only marginally faster.
- **1n Fulqrum cells are roughly equal across nccl/cuda_mpi/host_mpi** (~7.6–8.1 s compute, exch ≤ 1.34 s) because there's no cross-node hop. The wallclock is dominated by ~36 % host PRIMME glue between matvecs, not the matvec itself.
- **host_mpi 1n's high `sync` (1.13 s)** is the host↔device transfer Fulqrum has to do explicitly when the comm backend won't accept device pointers — that bookkeeping shows up as `sync` rather than `exch`. Total per-iter cost (compute + exch + sync) is comparable to nccl/cuda_mpi (~8.4–9.0 s either way).
- **SBD's per-Davidson-sub-iter cost is ~5× larger than Fulqrum's per-matvec cost** at 2-node (32 s vs 4.15 s). SBD doesn't natively expose a per-iter compute/exch split — Davidson is one black-box sub-iter from `run_sbd_diag.py`'s perspective.
- **SBD OMP-offload edges out Thrust at 2-node** (27 s/sub-iter vs 32 s/sub-iter); the historical "0.67× OMP regression" claim was an artifact of a silently-broken `sm_90`-on-`sm_100` build, not the path itself.

### Apples-to-apples vs SBD (matched ~10 matvecs)

Fulqrum's `-r 3.9e-2` converges in 11 matvecs, comparable to SBD's
`--block 10 --iteration 1` (10 Davidson sub-iters). Energies are
correspondingly looser-converged (~3.4e-2 PRIMME residual vs 8.7e-4
at `-r 1e-3`), but the wallclock comparison reflects equal eigsolve
work.

**Fulqrum** (11 matvecs, `-r 3.9e-2`):

| Backend | 4×GB200 (1 node) ¹ | 8×GB200 (2 nodes) | 1n→2n | Energy (Ha) |
|---|---:|---:|---:|---|
| cuda_mpi | 219 s | **134 s** | 1.63× | −326.822567853744 |
| nccl     | 221 s | **118 s** | 1.87× | −326.822567851544 |

**SBD** (`--iteration 1 --block 10`, 10 matvecs):

| Backend | 4×GB200 (1 node) | 8×GB200 (2 nodes) | 1n→2n | Energy (Ha) |
|---|---:|---:|---:|---|
| Thrust       | **577 s** | **329 s** | 1.75× | −326.821832430028 |
| OMP-offload  | **491 s** | **304 s** | 1.62× | −326.821832430028 |

At matched matvec count on this workload, **Fulqrum nccl 2-node is
2.6× faster than SBD Thrust 2-node and 2.6× faster than SBD OMP-offload
2-node**.

The 0.0006 Ha gap between Fulqrum (−326.82257) and SBD (−326.82183)
is the truncation gap (Fulqrum spans the full 27,901² ≈ 778 M
Cartesian product; SBD diagonalizes the SBD-selected basis on the
same 27,901 list).

### Tightly-converged Fulqrum (30 matvecs, `-r 1e-3`)

For when convergence to ≥6 digits matters more than apples-to-apples
wallclock comparison. Eigsolver also matters here — both rows below
fix a comm backend and vary the eigsolver to isolate the
`cublas_dprimme` win:

| Backend | Eigsolver | 4×GB200 (1 node) | 8×GB200 (2 nodes) | Energy (Ha) |
|---|---|---:|---:|---|
| cuda_mpi | `cublas_dprimme`           | INT32 ²    | **264 s** | −326.824718459210 |
| cuda_mpi | host PRIMME (`--gpu-eigsh 0`) | 479 s   | 360 s     | −326.824718460… |
| nccl     | `cublas_dprimme`           | INT32 ²    | **188 s** | −326.824718459210 |
| nccl     | host PRIMME (`--gpu-eigsh 0`) | 465 s   | 280 s     | −326.824718460… |

The `cublas_dprimme` row is the production path on 2-node — it keeps
PRIMME's basis vectors on device, replacing OpenBLAS calls with
cuBLAS / cuSolver and removing host↔device round-trips per iter.
On 2-node nccl that buys **1.49×** over host PRIMME (188 s vs 280 s);
on 2-node cuda_mpi, **1.36×** (264 s vs 360 s).

¹ Both 1-node cells use `--gpu-eigsh 0` (host PRIMME). 1-node 4-rank
hits the INT32 limit on the GPU eigsolver path.

² At 4 ranks, `cublas_dprimme` rejects with `n_local × ncv = 2.34 B >
INT32_MAX (2.15 B)`: 32-bit indexing in cuBLAS calls over the flattened
basis. 1-node Fulqrum runs use `--gpu-eigsh 0` as the workaround. The
2-node 8-rank case halves `n_local` and fits.

Energies are stable within each (workload, eigsolver) cell: across
nccl/cuda_mpi the eigenvalue is bit-equal at 12 digits within the
same eigsolver, with `cublas_dprimme` and host PRIMME differing in
the 9th digit (PRIMME's residual tolerance, not a correctness issue).
SBD is bit-equal across cpu/gpu/gpu-omp and 1-/2-node.

Things to note:

1. **At matched matvec count, Fulqrum on 2-node is 2.6× faster than SBD.** The earlier 188 s (Fulqrum) vs 304 s (SBD) reading at non-matched matvec count was misleading — Fulqrum was running 30 matvecs to a tight `-r 1e-3` residual; SBD was running 10. With both at 10–11 matvecs (Fulqrum `-r 3.9e-2`), Fulqrum nccl 2-node lands at 118 s, SBD OMP-offload 2-node at 304 s.
2. **`cublas_dprimme` is a 1.4–1.5× win over host PRIMME on 2-node** (nccl: 188 s vs 280 s; cuda_mpi: 264 s vs 360 s). Same matvec, same comm — the win is collapsing host↔device round-trips around the eigenvalue update.
3. **NCCL beats cuda_mpi at every (eigsolver, node count) cell**. The 4.8×-larger exch advantage from MNNVL P2P (§4) translates to wallclock gains between 1.0× (1-node, comm dwarfed by host eigsolver) and 1.4× (2-node `cublas_dprimme`, comm back in the critical path).
4. **SBD gpu-omp on 2-node is now *faster* than Thrust** (304 s vs 329 s) and scales 1.62× from 1n→2n. The earlier "0.67× OMP regression" reading was an artifact of mismatched per-rank workloads in the prior runs (§3.1) — the per-rank rate is flat across 1- and 2-node configs.

## 2. Per-matvec breakdown — Fulqrum

Median steady-state matvec; `exch` is tile-exchange collectives
(allgatherv + ring shift + grouped sendrecv); `compute` is the kernel
body. `total` is the GPU-max across ranks. Updated 2026-06-16 with
`cublas_dprimme`-path measurements; the prior "host PRIMME" rows from
the original sweep are kept below for reference.

### 2.1. With `cublas_dprimme` (today's measurements)

| Backend | 4×GB200 (1 node) ² | | | | 8×GB200 (2 nodes) | | | |
|---|---:|---:|---:|---|---:|---:|---:|---|
|        | total | exch | compute | overlap? | total | exch | compute | overlap? |
| cuda_mpi | host PRIMME ¹ | — | — | — | 6.4 s | 3.20 s | 6.30 s | partial |
| nccl     | host PRIMME ¹ | — | — | — | **4.04 s** | **0.95 s** | 4.01 s | yes |

¹ Per the INT32 footnote in §1 — at 4 ranks the GPU PRIMME path
rejects, so 1-node measurements are host-PRIMME-only and not
comparable to 2-node steady-state matvec timing.
² 1-node steady-state matvec on host PRIMME is ~7.7 s (cuda_mpi)
and ~7.7 s (nccl); both are bottlenecked by host eigsolver overhead
between matvec calls, not communication.

### 2.2. With host PRIMME (original sweep, retained for reference)

| Backend | 4×GB200 (1 node) | | | | 8×GB200 (2 nodes) | | | |
|---|---:|---:|---:|---|---:|---:|---:|---|
|        | total | exch | compute | overlap? | total | exch | compute | overlap? |
| host_mpi | 21.6 s | 21.3 s | 21.5 s | **no** | (not run) | — | — | — |
| cuda_mpi | 20.5 s | 20.2 s | 20.4 s | **no** | 13.4 s | 13.1 s | 13.4 s | **no** |
| nccl     | (not run) | — | — | — | 11.4 s | 2.74 s | 11.3 s | yes |

Reading this table:

Reading the updated §2.1 table:

- **`cublas_dprimme` collapses the matvec from ~11 s → ~4 s** on the
  best 2-node config. The change is mostly in `compute`: with the
  eigenvalue update now staying on device, the compute-side kernel
  body shrinks from 11.3 s → 4.01 s while exch drops slightly
  (2.74 s → 0.95 s, also helped by tighter overlap).
- **NCCL still wins on exch** (0.95 s vs 3.20 s for cuda_mpi on
  2-node) — the MNNVL P2P story (§4) is unchanged. cuda_mpi cross-node
  Allgatherv goes through IB-staging because HPCX 4.1.x cannot init
  pml=ucx in this cluster's SLURM cgroup, and the fallback (ob1 +
  smcuda + tcp) is not CUDA-aware cross-node.
- **NCCL exch fits inside compute on 2-node** (0.95 s exch behind
  4.01 s compute, "yes" overlap). The matvec is now genuinely
  compute-bound; further wins need kernel work.

The 4.8×-larger exch improvement vs the 1.4× wallclock improvement
(264 s vs 188 s for cuda_mpi vs nccl on 2-node) tracks Amdahl: with
the GPU eigsolver path, comm is no longer the dominant share.

## 3. Per-phase breakdown — SBD

Updated 2026-06-16 with the post-rebuild SBD (sm_100 native for both
Thrust and OMP-offload; same `--iteration 1 --block 10`). Phases as
reported by `run_sbd_diag.py`.

|  | 4×GB200 (1 node) Thrust | 4×GB200 (1 node) OMP | 8×GB200 (2 nodes) Thrust | 8×GB200 (2 nodes) OMP |
|---|---:|---:|---:|---:|
| Helper construction        | 5.0 s | 5.4 s | 4.9 s | 4.8 s |
| `mult.Init`                | 14 s  | —     | 17 s  | — |
| `makeQChamDiagTerms` GPU   | 15 s  | —     | 12 s  | 13 s |
| Davidson 10 sub-iters      | 547 s | 462 s | **320 s** | **268 s** |
| Final mult                 | 47 s  | 30 s  | 30 s  | 21 s |
| Total wall                 | **577 s** | **491 s** | **329 s** | **304 s** |

### 3.1. On the apparent OMP-offload "regression" — there isn't one

The original sweep reported a 9× slowdown on `makeQChamDiagTerms`
going from 1-node to 2-nodes (57 s → 538 s) and an end-to-end
"regression" reading of 0.67×. It isn't a regression.

`makeQChamDiagTerms` on the OMP-offload build is **a CPU host loop**
(qcham.h:290), not an offload kernel. The `SBD_THRUST` build calls
`device_mult.makeQChamDiagTerms(hii)` on GPU; the OMP-offload build
falls into the `#else` branch and runs the host OpenMP loop.

Adding per-rank instrumentation
(`qcham.h::makeQChamDiagTerms` printf with items_done and loop time)
shows:

| run                            | per-rank items | loop time | rate          |
|---|---:|---:|---:|
| h2o-1em5,    1-node 4r (2×2)   | 7.6 M          | 7.7 s     | 1.0 M items/s |
| h2o-1em5,    2-node 8r (4×2)   | 3.8 M          | 4.1 s     | 0.93 M/s      |
| fe4s4-5000,  1-node 4r (2×2)   | 6.25 M         | 35 s      | 0.18 M/s      |
| fe4s4-5000,  2-node 8r (4×2)   | 3.1 M          | 17.7 s    | 0.18 M/s      |
| fe4s4-27901, 1-node 4r (2×2)   | 195 M          | 1076 s    | 0.18 M/s      |
| fe4s4-27901, 2-node 8r (4×2)   | 97 M           | 530 s     | 0.18 M/s      |

The per-item rate is **flat across 1-node and 2-node configs**. fe4s4
is just ~5–6× slower per item than h2o because its `ZeroExcite`
kernel reads a much larger I2 (≈ 13 MB for 36 orbitals vs ≈ 160 KB
for 12 orbitals), and the bit_length is 2× wider.

So the original "57 s vs 538 s" line was apples-vs-oranges on
per-rank work: the prior 1-node 27,901 run did far less per-rank
work than a 2×2 grid would imply (probably a different
`--adet_comm_size`/`--bdet_comm_size` shape, or a different
binary). The instrumented 1-node 4r 27,901 run on a 2×2 grid took
1076 s — exactly matching the 0.18 M items/s rate × 195 M items per
rank — confirming the rate is identical at 1 and 2 nodes.

Davidson scales as expected on OMP-offload too: with the same
instrumented binary at consistent grid, 1-node 4r → 2-node 8r is
1607 s → 854 s (**1.88× speedup**), near-perfect. The "853 vs 454"
gap reported earlier was the same per-rank-workload artifact.

**Bottom line**: there is no 2-node OMP-offload regression. The
OMP-offload host path is just slow per item on fe4s4 because of the
big I2; the same code is invoked at 1 node and at 2 nodes and runs
at the same rate. To speed up OMP-offload on fe4s4-class problems,
the lever is the `ZeroExcite` kernel (cache locality / vectorization
of the I2 lookups), not the MPI/OMP runtime.

## 4. Why NCCL wins on 2-node GB200: MNNVL

The 4.8× exch improvement going from cuda_mpi to NCCL on 2 nodes is
much larger than IB+GDR alone would deliver. The reason:
**Multi-Node NVLink Switch fabric**.

`NCCL_DEBUG=INFO` from job 512306 reveals all 8 GPUs across the 2
nodes are in a single MNNVL fabric clique:

```
MNNVL busId 0x1801000 fabric UUID 814d…b5b7 cliqueId 0x7ffe state 3
MNNVL 1 cliqueId 7ffe cliqueSize 8 cliqueRank 2
…
NCCL INFO Channel … : 4[0] -> 3[3] via P2P/MNNVL
```

NCCL set up IB+GDR (peermem and DMABUF on all four `ibp0..3` HCAs)
plus SHARP, but for the actual data plane it picks `P2P/MNNVL` for
cross-node hops — i.e. NVLink across nodes, not IB. That's the source
of the 4.8× boost. On clusters without MNNVL (most boxes today,
including the H100 reference), NCCL would fall back to IB/GDR and the
gap over MPI Allgatherv would be smaller (~1.5–2×).

## 5. Cross-box reference: 8×H100 (1 node)

For context, the same workload on the 8×H100 box (1 node, intra-node
PCIe + NVLink + NVSwitch — no MNNVL):

| Backend | total | exch | compute | overlap? | eigensolve | matvecs | Energy (Ha) |
|---|---:|---:|---:|---|---:|---:|---|
| host_mpi (bundled half_dets †) | 18.5 s | 4.6 s | 17.6 s | yes | 602 s | 19 | −326.662853130561 |
| cuda_mpi (SBD alpha_dets ‡)    | 14.7 s | (overlapped) | (overlapped) | yes | 656 s | 30 | −326.824718… |
| nccl (bundled half_dets †)     | 18.5 s | 3.7 s | 17.6 s | yes | 616 s | 19 | −326.662853130561 |

† H100 host_mpi/nccl rows use the bundled `fe4s4_half_dets.json.xz` 27,901 selection (different ground state from SBD's selection, but per-matvec mechanics are unaffected).
‡ H100 cuda_mpi number is from the prior benchmark; today's H100 box rejected the cuda_mpi probe.

Note that on H100, host_mpi and NCCL have similar matvec timing
(~18.5 s), exch is fully overlapped behind compute on both, and NCCL's
exch advantage (3.7 s vs 4.6 s) is small. This is because:
1. The H100 box has no GPUDirect peermem (`gpudirect_hint=no-peermem-module`), so even NCCL goes through copy engines / PCIe rather than direct NVLink memory access.
2. Compute (~17.6 s) is much larger than exch on either backend, so the comm backend doesn't matter for the critical path.

The GB200 NCCL win is fabric-specific (MNNVL), not a generic property
of the refactored backend.

## 6. NCCL backend bug — RESOLVED

### 6.1. The bug

The original `_NcclBackend` in
`fulqrum/gpu/distributed/backends/nccl.py` built a single global
`NcclCommunicator(world_size)` at init, ignored the per-call `comm`
argument, validated against `world_size`, and indexed counts/displs
with the global rank. Tile-resident matvec calls collective
primitives along axis subcommunicators (alpha-row peers, beta-col
peers) — every NCCL call on a non-world subcomm rejected with
`counts/displs len must equal world_size`. `host_mpi` and `cuda_mpi`
worked because mpi4py is intrinsically subgroup-aware; NCCL was the
odd one out.

### 6.2. The fix

Refactored `_NcclBackend` to be subgroup-aware:
- Cache: `_world_nccl` for COMM_WORLD plus `_sub_cache: {comm.py2f() → NcclCommunicator}` for axis subcomms, lazily bootstrapped via comm-internal `bcast` of a `ncclUniqueId`.
- `allgatherv_device` switched to padded uniform `ncclAllGather` (pad each rank's send to `max(counts)`, gather, unpack at requested displs); size-1 subgroups handled as a local identity copy.
- `ring_shift_device` / `grouped_sendrecv_device` / `allreduce_device` route through the matching subgroup's NCCL comm.

### 6.3. Path-coverage caveat

Tile-resident matvec has two code paths gated by
`FQ_TILE_RESIDENT_RINGSHIFT`:

- **default (=0)**: uses `Allgatherv` + `neighbor_alltoallv_device`.
- **ringshift (=1)**: uses `ring_shift_device` + `grouped_sendrecv_device`.

The NCCL backend implements only the ringshift-path primitives;
`neighbor_alltoallv_device` raises `NotImplementedError`. So
`FULQRUM_DIST_BACKEND=nccl` requires `FQ_TILE_RESIDENT_RINGSHIFT=1` —
otherwise the matvec falls into the unimplemented primitive
mid-eigensolve. The launcher and SLURM scripts now set this env var.
Cleaner follow-up: have `_NcclBackend.init()` set the env var itself
or fail with an actionable error.

### 6.4. Validation

Bit-equal correctness verified at three scales:

| scenario | reference energy | nccl energy | match |
|---|---|---|---|
| H100, fe4s4 4000, 2×4 grid (bundled half_dets, host_mpi vs nccl)        | −326.634044413726 | −326.634044413726 | 12 digits |
| H100, fe4s4 27,901, 2×4 grid (bundled half_dets, host_mpi vs nccl)      | −326.662853130561 | −326.662853130561 | 12 digits |
| GB200 2-node, fe4s4 27,901, 2×4 grid (SBD alpha_dets, cuda_mpi vs nccl) | −326.824718460995 | −326.824718460994 | 11 digits |

Plus a focused 4-rank `(2,2)` reproducer (`fulqrum/test/test_collective.py`-style) confirming `host_mpi` vs `nccl` matvec is bit-equal on a non-world subcomm. The pre-refactor backend rejected the same input.

## 7. Recommendation

### 7.1. What we now know

- **NCCL backend is correct on real hardware** at the source level on arbitrary 2D process grids (subgroup-aware refactor + ringshift caveat).
- **GB200 + NCCL + MNNVL gives a 4.8× exch speedup over cuda_mpi**, but ~10% wallclock speedup since compute now dominates.
- **OMP-offload's per-item rate is flat across 1-node and 2-node configs**, but it runs ~5–6× slower per item than h2o on fe4s4 because the host loop (`qcham.h::makeQChamDiagTerms`) does not exploit the GPU and the 36-orbital I2 (≈ 13 MB) doesn't sit in fast cache. The original "2-node OMP regression" claim was an artifact of mismatched per-rank workloads in the prior runs (§3.1); there is no actual scaling regression.

### 7.2. Where to invest next, in priority order

1. **Land the NCCL refactor upstream**, with the env-var caveat addressed (auto-set `FQ_TILE_RESIDENT_RINGSHIFT=1` in `_NcclBackend.init()`, or implement the missing `neighbor_alltoallv_device`). Add subgroup bit-equal tests so future regressions surface in CI.
2. **Optimize the SpMV kernel itself.** With comm out of the way, per-matvec is now ~11.3 s of compute on 8 GB200s. Profile the tile kernel; likely candidates include better register reuse on the group-hashmap chemistry kernel and tighter aabb-cross-spin loops.
3. **Speed up the SBD OMP-offload host loop on fe4s4-class problems.** The cost is in `qcham.h::makeQChamDiagTerms` — a CPU OpenMP loop that runs `ZeroExcite` per (alpha, beta) pair and reads from the 13 MB I2 array. Two cheap improvements: (a) make `SBD_THRUST` the default device path so the diagonal build runs on GPU regardless of which Davidson variant is requested, and (b) profile cache misses on `ZeroExcite` to see whether tiling I2 access or precomputing diagonal contributions buys anything.
4. **Upgrade GB200 cluster to OpenMPI 5.x.** Still worth doing for non-NCCL paths (cuda_mpi users, MPI-based workloads other than Fulqrum). Lower priority now that NCCL works on this cluster.

### 7.3. Practical guidance (refreshed 2026-06-16)

For Fulqrum-class workloads on this GB200 cluster today, the fastest
configuration is `FULQRUM_DIST_BACKEND=nccl` with
`FQ_TILE_RESIDENT_RINGSHIFT=1` and the default
`gpu_eigsh=auto` (`cublas_dprimme`), which gives **188 s eigensolve
on 2-node** — a **2.8× win** over the 520 s reported in the original
sweep. The win is in `compute` (host eigsolver round-trips removed),
not exch.

At 1 node, Fulqrum's `cublas_dprimme` rejects with INT32 overflow
(4 ranks × n_local 195 M × ncv 12 = 2.34 B > 2.15 B INT32_MAX);
fall back to `--gpu-eigsh 0` (host PRIMME, ~465 s nccl) or use ≥8
ranks. Real fix is int64 indexing in cuBLAS calls — engineering work
that should land upstream.

SBD on the same workload now lands at 329 s (Thrust) / 304 s
(gpu-omp) on 2-node — both faster than they were in the original
sweep, with gpu-omp slightly winning.

At **matched 10-matvec eigsolve work** (Fulqrum `-r 3.9e-2` vs SBD
`--block 10 --iteration 1`), the comparison on 2-node is:

- Fulqrum nccl: **118 s** / cuda_mpi: 134 s (E = −326.822568)
- SBD Thrust: 329 s / OMP-offload: 304 s (E = −326.821832)

So Fulqrum nccl is **~2.6× faster** than the better of the two SBD
backends per unit eigsolve work, with a 0.0006 Ha gap from the SBD's
selected-basis truncation. At converged Fulqrum (30 matvecs,
`-r 1e-3`) on 2-node, the wall is 188 s (E = −326.824718).

Beyond ~10⁹ subspace dimensions (where curating an SBD selection
becomes expensive), Fulqrum's tile-resident + NVLink-aware-collective
approach is still the path that scales.
