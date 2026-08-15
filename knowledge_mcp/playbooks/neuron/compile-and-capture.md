---
id: neuron/compile-and-capture
chip: neuron
title: Compiling to NEFF and capturing a Neuron profile
tags: [neuronx-cc, neff, ntff, neuron-profile, capture, hlo, compile]
refs:
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/
---
The Neuron profiling artifacts are a **NEFF** (the compiled executable) and an **NTFF** (the runtime
trace captured while it runs). The analysis MCP reads both off the mount; this playbook is how they
are produced.

**Compile (NEFF).** `neuronx-cc` is a device-free compiler — it does not need a Trainium/Inferentia
device, which is why compile-only hosts work. The device dependence people hit is in the *framework
lowering*, not the compiler: `torch_neuronx.trace` wants a device to trace the graph. To build a NEFF
on a plain CPU box, lower the model to HLO without a device (e.g. `PJRT_DEVICE=CPU` with the XLA HLO
dump) and then `neuronx-cc compile --framework XLA --target <trn> <graph.hlo.pb> -o model.neff`. The
compile itself needs only `neuronx-cc`; the Neuron runtime library is an *execution* dependency
(and, in practice, an import dependency of `torch_neuronx` during lowering) — not something
`neuronx-cc` requires to compile an HLO proto.

**Capture (NTFF).** Capturing a runtime trace **requires a device** and the same NeuronCore
configuration you serve with (including the logical-NeuronCore setting — see
`neuron/dma-and-collectives`). Run the capture on a dedicated or whole-device allocation, not a
contended shared core, and write the resulting `.ntff` to durable storage as part of the same run so
it is not lost when the capture process exits.

**Analyze (device-free).** Post-processing the NEFF + NTFF into a per-core report needs **no
device**, which is why the accelprof analysis MCP runs it on a CPU pod — its `neuron-summary` analyzer is
the concrete command (a text report of engine-active times and DMA). Machine-readable and timeline
formats also exist; check your installed `aws-neuronx-tools` version for the exact subcommand and
`--output-format` options rather than assuming them. Start from the text/JSON summary and drop to a
timeline only when you need per-op ordering.
