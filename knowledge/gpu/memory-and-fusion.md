---
id: gpu/memory-and-fusion
chip: gpu
title: Memory-bound elementwise, coalescing, and epilogue fusion
tags: [fusion, epilogue, memory-bound, coalescing, bank-conflict, kernel-launch, elementwise]
refs:
  - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
  - https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
---
Elementwise ops (bias, activation, residual add, dropout, layernorm tails) have near-zero
arithmetic intensity: each reads and writes the whole tensor, so they are memory-bound and their
cost is bandwidth, not math.

**Symptom:** a GEMM near peak Tensor-Core throughput, followed by separate bias / activation
kernels that are each DRAM-bandwidth-bound, and an end-to-end time far above the compute-only
estimate. In the trace you see several kernels each moving the full output tensor.

**Cause:** the output tensor is read and written multiple times (GEMM writes it, bias reads+writes,
activation reads+writes), and each separate launch adds kernel-launch overhead.

**Fix — fuse into the producer's epilogue.** Fold bias, activation, and residual into the GEMM
epilogue (CUTLASS epilogue, or a compiler fusion pass / `torch.compile` region) so the result is
written to DRAM **once**. The memory-bound kernels disappear and the step approaches the GEMM's
compute bound. Note: shared memory is block-scoped scratch — it cannot pass data between separate
kernel launches, so "use shared memory" is not a substitute for fusion.

**Access pattern.** For custom kernels, uncoalesced or large-stride global access inflates the
sectors transferred per warp request; the wasted bandwidth caps throughput even at high occupancy.
Confirm with *sectors per request* in Nsight Compute. Shared-memory stride that shares banks causes
bank conflicts; conflict degree is `32 / gcd(stride, 32)` distinct banks — a stride coprime to 32
(e.g. odd) avoids it.

**Watch for graph breaks (`torch.compile`).** Frequent Python-level branches cause graph breaks
that split the graph into small subgraphs; cross-op fusion cannot cross a break and eager overhead
returns at each boundary. Reduce breaks around hot regions so the fusable window is large.
