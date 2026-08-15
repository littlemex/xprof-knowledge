---
id: neuron/compile-and-capture
chip: neuron
title: Compiling to NEFF and capturing a Neuron profile
tags: [neuronx-cc, neff, ntff, neuron-profile, neuron-explorer, capture, hlo]
refs:
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/
---
The Neuron profiling artifacts are a **NEFF** (the compiled executable) and an **NTFF** (the runtime
trace captured while it runs). The analysis MCP reads both off the mount; this playbook is how they
are produced.

**Compile (NEFF).** `neuronx-cc` compiles a framework graph to a `.neff`. To produce a NEFF without a
Trainium device (CI, a CPU box), lower the model to HLO first (e.g. `PJRT_DEVICE=CPU` with the XLA
HLO dump) and compile the HLO proto with `neuronx-cc compile --framework XLA --target <trn> ... -o
model.neff`. A device-free compile image needs the Neuron runtime lib and matching Python/libarchive
present, but no accelerator.

**Capture (NTFF).** `neuron-profile capture -n model.neff -s profile.ntff` runs the NEFF on a
NeuronCore and records the trace; this step **requires a device** and the same core configuration
you intend to serve with. Guard against capturing on a busy shared core (LNC mismatch) — request a
whole-device or dedicated allocation. Job-based captures can be reaped before you copy the `.ntff`
out; write it to durable storage (or the trace bucket) as part of the same job.

**Analyze (device-free).** `neuron-explorer view -n model.neff -s profile.ntff --output-format
summary-text` prints the per-core engine-active / MFU / DMA report and needs **no device** — which is
why the analysis MCP runs it on a CPU pod. Other formats: `summary-json` (machine-readable),
`perfetto` (open the timeline in the Perfetto UI), `db` (a queryable server). Start from
`summary-text`/`summary-json` and drop to the timeline only when you need per-op ordering.
