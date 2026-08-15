---
id: gpu/roofline
chip: gpu
title: Roofline diagnosis — compute-bound vs memory-bound vs latency-bound
tags: [roofline, memory-bound, compute-bound, latency-bound, occupancy, nsight-compute, ncu]
refs:
  - https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline
  - https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
---
The roofline places a kernel by its **arithmetic intensity** (FLOPs per byte): to the left of the
ridge point it is bounded by memory bandwidth, to the right by peak compute. Use it to decide which
resource to optimize — never optimize the one that is not the limiter.

**Symptom → reading**
- **Point sits ON the memory roofline (low intensity):** memory-bound. Raise intensity (fuse ops so
  bytes are moved once — see `gpu/memory-and-fusion`), or cut bytes (lower precision, better
  layout). Adding compute or occupancy will not help.
- **Point sits ON the compute roofline (high intensity):** compute-bound and near-optimal. Gains
  come from a better algorithm, Tensor Cores, or lower-precision math (see `gpu/tensor-cores`).
- **Point sits BELOW both rooflines:** latency-bound — the kernel is neither saturating bandwidth
  nor the math units. This is the most misread case.

**The occupancy trap.** A kernel can show 90%+ achieved occupancy and still sit far below both
rooflines. Occupancy only means enough warps are resident to *potentially* hide latency; it does
not guarantee the memory system or issue pipeline is used well. Common real causes when occupancy
is high but throughput is low: uncoalesced or large-stride global loads (many sectors per request),
shared-memory bank conflicts, or long dependency chains that stall issue. Confirm in Nsight
Compute's **Memory Workload Analysis** — look at *sectors per request* (coalescing) and L2/L1 hit
rates — before touching launch configuration.

**What to try, in order:** (1) fuse to remove memory-bound elementwise passes; (2) fix the access
pattern (coalescing/layout) if sectors-per-request is high; (3) only then revisit occupancy/tiling,
and only if the roofline says you are actually compute- or bandwidth-limited.
