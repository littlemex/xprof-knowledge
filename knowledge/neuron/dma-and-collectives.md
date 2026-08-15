---
id: neuron/dma-and-collectives
chip: neuron
title: DMA stalls and collective-communication bottlenecks on Neuron
tags: [dma, collective, cc, allreduce, overlap, tensor-parallel, lnc, bandwidth-bound]
refs:
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/
---
When the engine-active breakdown shows the compute engines idle but wall time high, the limiter is
usually data movement — either on-chip/off-chip DMA or an inter-core collective.

**DMA-bound.** Weights or activations are not staged into SBUF/PSUM ahead of the compute that needs
them, so the PE engine waits on the DMA queues. In the profile you see DMA activity filling the gaps
where PE is idle. Levers: increase reuse so fewer bytes move (batching, tiling), overlap DMA with
compute (double-buffering the next tile while the current one computes), and check the compiler is
not spilling due to an oversized working set.

**Collective-bound (tensor/sequence parallel).** All-reduce / all-gather / reduce-scatter between
NeuronCores serializes with compute if it is not overlapped. Symptom: periodic gaps aligned to layer
boundaries where all compute engines idle while a collective runs. Levers: overlap the collective
with independent compute (the compiler/runtime can, if the graph exposes the independence), size the
parallelism so the collective payload is worth the launch, and confirm the inter-core fabric (not
compute) is the bottleneck before scaling out further.

**Logical NeuronCore (LNC) and placement.** Capturing or serving on the wrong core count or a
contended device skews the picture — a whole-device capture on a shared node can mismatch the LNC
configuration and produce a misleading or failed profile. Profile on a dedicated allocation with the
same core configuration you serve with, so the collective and DMA timings reflect production.
