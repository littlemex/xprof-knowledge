---
id: common/metrics-glossary
chip: common
title: Metrics glossary (serving and profiling)
tags: [glossary, ttft, tpot, itl, mfu, mbu, hfu, throughput, latency]
refs:
  - https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
---
Shared vocabulary so a metric means the same thing across GPU and Neuron runs.

**Serving latency**
- **TTFT** (time to first token): prefill + queue + any KV transfer, measured at the client on the
  first streamed token. Dominated by prompt length and scheduling.
- **TPOT** (time per output token): steady-state decode time per token, `(E2E − TTFT)/(out−1)`
  (some tools divide by `out` — note which, as they differ for short outputs). Dominated by
  decode-step efficiency and memory bandwidth.
- **ITL** (inter-token latency): the per-token gaps; TPOT is their mean, but the p99 exposes jitter.
- **E2E**: total request latency. **request/output/total throughput**: requests·s⁻¹ and tokens·s⁻¹.

**Utilization (how close to the hardware limit)**
- **MFU** (model FLOPs utilization): useful model FLOPs ÷ peak FLOPs. Use the **dtype-matched, dense**
  peak (the BF16/FP8 number for a BF16/FP8 run, and never the 2× sparsity figure) — mixing dtypes or
  the sparse peak makes MFU meaningless. Low MFU on a compute-bound step means the math units are
  starved.
- **MBU** (model bandwidth utilization): achieved bytes·s⁻¹ ÷ peak bandwidth. Compare against the
  **achievable** bandwidth (~70–90% of the spec figure), not the datasheet peak, or a bandwidth-bound
  kernel looks artificially far from 100%. High MBU with low MFU is the signature of a memory-bound
  step (typical of decode).
- **HFU** (hardware FLOPs utilization): includes FLOPs the model does not strictly need; HFU > MFU
  means work is redone or wasted — recomputation (activation checkpointing) or padding/masked compute.
- **Arithmetic intensity**: FLOPs per byte moved; where a kernel sits on it decides compute- vs
  memory-bound (see `gpu/roofline`).

**Occupancy** (GPU): resident warps ÷ max warps. Distinguish **theoretical** occupancy (the limit
set by registers/shared memory/block size) from **achieved** occupancy (measured at runtime). It is
a necessary condition for latency hiding, **not** a sufficient one for efficiency — high occupancy
with low throughput points at memory access, not warp count (see `gpu/roofline`,
`gpu/memory-and-fusion`).
