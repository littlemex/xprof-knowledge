# xprof-knowledge — cross-accelerator profiling & tuning knowledge over MCP

A single MCP that serves the profiling and tuning know-how for **NVIDIA GPU**, **AWS Neuron**, and
the cross-cutting methodology — so an agent driving the profile → analyze → improve → re-experiment
loop retrieves the relevant playbook remotely, instead of relying on local skills that only exist on
one machine.

The content is **curated, distilled playbooks** — symptom → likely cause → what to check → what to
try, with links to the upstream vendor documentation. It is deliberately not a copy of the vendor
docs and not a retrieval stack over them: the corpus is small and hand-written, so a transparent
keyword search beats an embedding index we would have to reinvent and keep fresh.

It is one of three single-responsibility MCPs an agent connects to: the **MLflow MCP** searches
experiments and runs, the **[xprof](https://github.com/littlemex/xprof)** analysis MCP resolves a
run to its profile files and analyzes them, and this MCP supplies the diagnosis knowledge.
Deployment (the CPU Pod, MCP hosting) lives in the **distributed-ai** repository.

## Tools and resources

- `list_topics(chip="")` — the available playbooks (`gpu` | `neuron` | `common`), id + title + tags.
- `get_topic(topic_id)` — one playbook in full (body + reference links), e.g. `gpu/roofline`.
- `search_knowledge(query, chip="")` — the playbooks most relevant to a symptom, keyword-ranked
  (title/tag hits weigh more than body), e.g. `search_knowledge("memory bound but occupancy high", "gpu")`.
- Resource template `knowledge://{chip}/{slug}` — each playbook as a browsable MCP resource.
- Prompt `diagnose(chip, symptom)` — primes an agent to search → read → propose one change.

## Content layout

```
knowledge/
  common/   methodology, metrics glossary
  gpu/      roofline, memory-and-fusion, tensor-cores-and-occupancy
  neuron/   utilization (MFU/MBU/HFU), dma-and-collectives, compile-and-capture
```

Each file is Markdown with a small front matter (`id`, `chip`, `title`, `tags`, `refs`). **Add a
playbook** by dropping a new `.md` under the right chip directory — no code change; it is picked up
at boot and becomes searchable and browsable.

## Testing

```bash
pip install -r knowledge_mcp/requirements.txt pytest
python -m pytest knowledge_mcp/ -q
```

Tests cover front-matter parsing, list/get/search, and that every packaged playbook loads and is
well-formed. The server runs on a CPU Pod with no accelerator and no mount, reached over
`kubectl port-forward` as a streamable-http MCP.
