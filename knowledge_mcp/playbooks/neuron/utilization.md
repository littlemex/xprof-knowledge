---
id: neuron/utilization
chip: neuron
title: Reading NeuronCore utilization — MFU/MBU/HFU and engine-active time
tags: [mfu, mbu, hfu, tensor-engine, vector, scalar, engine-active, arithmetic-intensity]
refs:
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/
---
A NeuronCore (v2/v3, on Trn1/Trn2/Inf2) runs several engines concurrently — the Tensor engine
(matmul), the Vector and Scalar engines (elementwise/activation/reduction), the GP-SIMD engine, plus
the DMA paths and the sync/collective path. The first diagnosis question is which engine is the
limiter, from the **engine-active** breakdown in the per-core profile report. The accelprof analysis MCP
produces that report via its `neuron-summary` analyzer; confirm the exact engine row names against
your own profile, since the label set varies by NeuronCore generation and tool version.

**Read the engine-active breakdown in this order**
- **Tensor engine active% high, others low → compute-bound on matmul.** Good place to be; further
  gains need a better algorithm or precision, not scheduling. Check MFU is actually high (below).
- **Vector/Scalar active% dominates → the elementwise/activation/reduction work is the limiter,** not
  the matmul. Fuse or reduce those ops, or overlap them with Tensor-engine work.
- **All engines low, wall time high → stalled,** usually on DMA or a collective (see
  `neuron/dma-and-collectives`), or on host/framework overhead between executions.

**MFU vs MBU vs HFU** (definitions in `common/metrics-glossary`). The profile gives you engine-active
time and DMA directly; **MFU/MBU usually have to be computed** — achieved FLOPs (or bytes) over the
dtype-matched peak × time — rather than read off the summary. Low MFU with high MBU is
memory-bandwidth-bound (typical of token-by-token decode and small batch); low MFU *and* low MBU is
stall/overhead-bound. **HFU noticeably above MFU** means the hardware does real FLOPs the model does
not need — recomputation, or padding/masked work — so reduce recomputation or wasted padding.

**Arithmetic intensity** governs the ceiling exactly as on GPU: a low-intensity step (small batch,
skinny matmul, decode) is bandwidth-bound and will not approach peak PE FLOPs — raise batch or fuse
to increase bytes-reused-per-load rather than chasing PE utilization.
