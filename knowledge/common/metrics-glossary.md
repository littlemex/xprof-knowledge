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
- **TPOT** (time per output token): steady-state decode time per token, `(E2E − TTFT)/(out−1)`.
  Dominated by decode-step efficiency and memory bandwidth.
- **ITL** (inter-token latency): the per-token gaps; TPOT is their mean, but the p99 exposes jitter.
- **E2E**: total request latency. **request/output/total throughput**: requests·s⁻¹ and tokens·s⁻¹.

**Utilization (how close to the hardware limit)**
- **MFU** (model FLOPs utilization): achieved FLOPs ÷ peak FLOPs. Low MFU on a compute-bound step
  means the math units are starved.
- **MBU** (model bandwidth utilization): achieved bytes·s⁻¹ ÷ peak bandwidth. High MBU with low MFU
  is the signature of a memory-bound step (typical of decode).
- **HFU** (hardware FLOPs utilization): includes recomputation; HFU > MFU means work is being redone
  (e.g. activation recomputation).
- **Arithmetic intensity**: FLOPs per byte moved; where a kernel sits on it decides compute- vs
  memory-bound (see `gpu/roofline`).

**Occupancy** (GPU): resident warps ÷ max warps. A necessary condition for latency hiding, **not**
a sufficient one for efficiency — high occupancy with low throughput points at memory access, not
warp count (see `gpu/roofline`, `gpu/memory-and-fusion`).
