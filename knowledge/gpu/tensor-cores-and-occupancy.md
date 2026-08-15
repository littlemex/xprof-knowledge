---
id: gpu/tensor-cores-and-occupancy
chip: gpu
title: Tensor Core utilization and when occupancy is the wrong lever
tags: [tensor-core, mfu, occupancy, precision, gemm, tail-effect, wave-quantization]
refs:
  - https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
  - https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
---
For matmul-heavy steps the question is whether the Tensor Cores are actually fed.

**Low MFU on a GEMM you believe is compute-bound.** Check, in order:
- **Precision / instruction path.** Is the math running on Tensor Cores at all? FP16/BF16/FP8 inputs
  with the right layout hit the TC path; an accidental FP32 path or an unsupported shape falls back
  to CUDA cores at a fraction of the throughput.
- **Shape and tiling — wave quantization.** If the problem does not tile evenly over SMs, the last
  wave runs with idle SMs (the tail effect); throughput drops even though each tile is efficient.
  Round M/N to the tile size, or pick a GEMM algo whose tiling fits the SM count.
- **K too small / memory-bound GEMM.** A skinny GEMM (small K, or batch-1 GEMV in decode) has low
  arithmetic intensity and is bandwidth-bound — it will never reach the compute roofline; treat it
  as memory-bound (see `gpu/roofline`).

**When NOT to raise occupancy.** If the GEMM is already at ~90% of peak TC throughput, occupancy is
not the limiter — more resident warps do nothing. Likewise, `__syncthreads()` correctness is about
the whole block reaching the barrier, unrelated to occupancy; a branch that lets only some warps hit
the barrier deadlocks regardless of how the branch aligns to the warp size. Raising occupancy helps
only when the roofline shows latency-bound behavior *and* the stall is from too few warps to hide
memory latency — verify with the scheduler/stall reasons in Nsight Compute before changing the
launch config.
