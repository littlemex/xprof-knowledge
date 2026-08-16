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
  with the right layout and alignment (16-byte, and K/N a multiple of 8–16) hit the TC path; an
  unaligned or unsupported shape, or FP32 without TF32 enabled (`torch.backends.cuda.matmul.allow_tf32`),
  falls back to a slower path. Confirm the TC pipe is active
  (`sm__inst_executed_pipe_tensor_op...pct_of_peak_sustained_active`, or DCGM TensorActive) — note
  that is an *issue* rate, not efficiency.
- **Tile quantization.** If M/N are not multiples of the GEMM tile, each tile wastes the padded
  edge. Round M/N to the tile size.
- **Wave quantization.** If the number of tiles is not a multiple of the SM count, the last wave
  runs with idle SMs (the tail effect) even though each tile is efficient. Pick a GEMM algo whose
  tiling fits the SM count, or resize so tiles fill whole waves.
- **K too small / memory-bound GEMM.** A skinny GEMM (small K, or batch-1 GEMV in decode) has low
  arithmetic intensity and is bandwidth-bound — it will never reach the compute roofline; treat it
  as memory-bound (see `gpu/roofline`).

**When NOT to raise occupancy.** If the GEMM is already at ~90% of peak Tensor-Core throughput,
occupancy is not the limiter — more resident warps do nothing. Raising occupancy helps only when
the roofline shows latency-bound behavior *and* the stall is from too few warps to hide memory
latency — verify with the scheduler/stall reasons in Nsight Compute before changing the launch
config.
