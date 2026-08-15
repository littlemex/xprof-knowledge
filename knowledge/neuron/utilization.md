---
id: neuron/utilization
chip: neuron
title: Reading NeuronCore utilization — MFU/MBU/HFU and engine-active time
tags: [mfu, mbu, hfu, pe, activation, pool, engine-active, arithmetic-intensity, neuron-explorer]
refs:
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/
---
A NeuronCore is several engines that run concurrently — the PE array (matmul/tensor), the
Activation and Pool/Vector engines, and the DMA/SP paths. The first diagnosis question is which
engine is the limiter, from the **engine-active** breakdown in `neuron-explorer view` (the
`summary-text`/`summary-json` report the analysis MCP surfaces via the `neuron-summary` analyzer).

**Read the summary in this order**
- **PE active% high, others low → compute-bound on matmul.** Good place to be; further gains need a
  better algorithm or precision, not scheduling. Check MFU is actually high (below).
- **Activation/Pool active% dominates → the elementwise/vector work is the limiter,** not the
  matmul. Fuse or reduce those ops, or overlap them with PE work.
- **All engines low, wall time high → stalled,** usually on DMA or a collective (see
  `neuron/dma-and-collectives`), or on host/framework overhead between executions.

**MFU vs MBU vs HFU** (definitions in `common/metrics-glossary`): low MFU with high MBU is
memory-bandwidth-bound (typical of token-by-token decode and small batch); low MFU *and* low MBU is
stall/overhead-bound. **HFU noticeably above MFU** means the graph is recomputing — e.g. activation
recomputation — so the hardware does real FLOPs that the model does not need; reduce recomputation
or checkpoint less aggressively if memory allows.

**Arithmetic intensity** governs the ceiling exactly as on GPU: a low-intensity step (small batch,
skinny matmul, decode) is bandwidth-bound and will not approach peak PE FLOPs — raise batch or fuse
to increase bytes-reused-per-load rather than chasing PE utilization.
