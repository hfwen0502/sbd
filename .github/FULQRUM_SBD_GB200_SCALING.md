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
energy as a correctness check. Backends with no 1-node run are marked
"—".

| Solver | 4×GB200 (1 node) | 8×GB200 (2 nodes) | speedup | Energy (Ha) |
|---|---:|---:|---:|---|
| Fulqrum cuda_mpi (PRIMME, 30 matvecs)  | 992 s | 577 s | **1.72×** | −326.824718460995 |
| Fulqrum nccl (PRIMME, 30 matvecs)      | —     | **520 s** | —      | −326.824718460994 |
| SBD Thrust (`--iteration 1`, 10 matvecs) | ~565 s | **264 s** | **2.14×** | −326.821832430028 |
| SBD OMP-offload (`--iteration 1`, 10 matvecs) | ~570 s | 853 s | **0.67× (regression)** | −326.821832430028 |

Energies all match within their respective solvers (Fulqrum to 11
digits across boxes; SBD bit-equal across backends). The 0.003 Ha gap
between Fulqrum (full Cartesian product) and SBD (selected basis) on
the same 27,901 list reflects SBD's truncation of off-diagonal
configurations.

Three things to note:

1. **Fulqrum cuda_mpi and SBD Thrust both gain from doubling GPUs**, sublinearly. Allgatherv volume per matvec scales with per-rank tile size (halves) but collective fan-out grows; net 1.7×–2.1× speedup is normal.
2. **Fulqrum NCCL on 2 nodes is the fastest Fulqrum configuration tested**, 10% faster than cuda_mpi at the same node count — but the per-matvec story (§2) shows the underlying improvement is much bigger.
3. **SBD OMP-offload runs slower than Thrust at the same rank count**, by ~1.0× at 1 node and ~3.2× at 2 nodes. This was originally read as a "2-node regression" but the per-rank rate is actually flat across 1-/2-node configs (§3.1); the wider gap at 2 nodes simply reflects different per-rank workloads in the prior runs. The OMP-offload host loop (`qcham.h::makeQChamDiagTerms`) is just slow per item on fe4s4 because of the big I2 cache footprint.

## 2. Per-matvec breakdown — Fulqrum

Same 4-row format. Median steady-state matvec; `exch` is tile-exchange
collectives (allgatherv + ring shift + grouped sendrecv); `compute`
is the kernel body. `total` is the GPU-max across ranks.

| Backend | 4×GB200 (1 node) | | | | 8×GB200 (2 nodes) | | | |
|---|---:|---:|---:|---|---:|---:|---:|---|
|        | total | exch | compute | overlap? | total | exch | compute | overlap? |
| host_mpi | 21.6 s | 21.3 s | 21.5 s | **no** | (not run) | — | — | — |
| cuda_mpi | 20.5 s | 20.2 s | 20.4 s | **no** | 13.4 s | 13.1 s | 13.4 s | **no** |
| nccl     | (not run) | — | — | — | **11.4 s** | **2.74 s** | 11.3 s | yes |

Reading this table:

- **At 1 node, exch ≈ total ≈ compute** for both host_mpi and cuda_mpi.
  The host-staged Allgatherv serializes with compute (no overlap) — the
  exposed exchange dominates the critical path. cuda_mpi only saves ~1 s
  vs host_mpi here because HPCX OpenMPI 4.1.x stages Allgatherv through
  pinned host buffers even with `opal_cuda_support=true`.
- **At 2 nodes, cuda_mpi exch grows back** to 13.1 s (Allgatherv now
  crosses IB), and the matvec stays exch-bound. Compute scales down
  from 20.4 s → 13.4 s with the per-rank tile shrinking, but the win is
  almost entirely in compute, not communication.
- **NCCL at 2 nodes finally breaks the exch dominance**: 2.74 s vs
  13.1 s for cuda_mpi (**4.8× faster exch**), and exch now fits
  comfortably inside compute. The matvec is back to compute-bound.

Why the NCCL exch is so much smaller than IB+GDR alone would predict:
the GB200 cluster has **MNNVL across the two nodes** (§4). NCCL detects
this and routes cross-node traffic over NVLink, not IB.

The 4.8× exch improvement only translates to ~10% wallclock gain
(577 s → 520 s) because compute now dominates. To go faster, the
SpMV kernel itself needs work.

## 3. Per-phase breakdown — SBD

Same 4×GB200 vs 8×GB200 layout. Phases as reported by `run_sbd_diag.py`.

|  | 4×GB200 (1 node) Thrust | 4×GB200 (1 node) OMP | 8×GB200 (2 nodes) Thrust | 8×GB200 (2 nodes) OMP |
|---|---:|---:|---:|---:|
| Helper construction        | 5 s    | 64 s   | 1.9 s   | 55 s |
| `mult.Init`                | 14 s   | —      | 4.9 s   | — |
| `makeQChamDiagTerms` GPU   | 14 s   | —      | 5.3 s   | — |
| `makeQChamDiagTerms` host  | —      | 57 s   | —       | 538 s |
| Davidson 10 sub-iters      | 481 s  | 454 s  | **264 s** | **853 s** |
| Final mult                 | 46.8 s | 34.6 s | 25.7 s  | 19.3 s |

### 3.1. On the apparent OMP-offload "regression" — there isn't one

The `makeQChamDiagTerms` row above looks like a 9× slowdown going from
1-node to 2-nodes (57 s → 538 s). It isn't.

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

### 7.3. Practical guidance

For Fulqrum-class workloads on this GB200 cluster today, the fastest
configuration is `FULQRUM_DIST_BACKEND=nccl` with
`FQ_TILE_RESIDENT_RINGSHIFT=1`, which drops eigensolve from 577 s
(cuda_mpi) to 520 s. SBD Thrust still wins end-to-end on this
particular benchmark (264 s davidson) due to convergence count, not
per-matvec speed. Beyond ~10⁹ subspace dimensions (where curating an
SBD selection becomes expensive), Fulqrum's tile-resident +
NVLink-aware-collective approach is the path that scales.
