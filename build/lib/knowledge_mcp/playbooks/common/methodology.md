---
id: common/methodology
chip: common
title: The profile → analyze → improve → re-experiment loop
tags: [methodology, loop, workflow, regression, roofline, bottleneck]
refs:
  - https://docs.nvidia.com/nsight-systems/UserGuide/index.html
  - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/
---
Performance work is a loop, not a one-shot measurement. Each turn changes exactly one thing so the
next profile is attributable.

1. **Measure the whole first.** Capture an end-to-end trace (nsys for GPU, neuron-profile for
   Neuron) before zooming into a kernel. Most wins come from removing work or overlap gaps, not from
   micro-optimizing an already-small kernel.
2. **Classify the bottleneck.** Is the step compute-bound, memory-bound, latency/overhead-bound, or
   communication-bound? The roofline (see `gpu/roofline`) and the engine-active breakdown (see
   `neuron/utilization`) tell you which. Do not optimize a dimension that is not the limiter.
3. **Form one hypothesis and change one variable.** "Fusing bias+activation removes two
   memory-bound kernels" is testable; "make it faster" is not.
4. **Re-run and compare like-for-like.** Same shapes, same batch, same warmup. Log both runs to the
   experiment store under the same alias with an incrementing `run_no` so the two are directly
   comparable, and keep the raw profile as an artifact.
5. **Stop when the limiter moves.** Once a step reaches its roofline (or the limiter shifts to a
   different resource), further work there has diminishing returns — re-classify and move on.

How this maps to the tooling: the **MLflow MCP** finds and compares runs; the **analysis MCP**
(accelprof) resolves a run to its profile files on the mount and runs the profiler; this **knowledge
MCP** supplies the diagnosis playbooks. A typical turn: search MLflow for the run → analyze its
trace → search this knowledge base for the symptom → apply one change → log the next run.
