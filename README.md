# xprof-knowledge — cross-accelerator profiling & tuning knowledge over MCP

A self-contained MCP server that serves profiling and tuning know-how for **NVIDIA GPU**, **AWS
Neuron**, and the cross-cutting methodology — so an agent (or you) driving the profile → analyze →
improve → re-experiment loop can retrieve the relevant playbook over MCP.

It depends on nothing but Python: no accelerator, no cloud account, no Kubernetes, no external
service. `pip install`, run one command, connect any MCP client.

The content is **curated, distilled playbooks** — symptom → likely cause → what to check → what to
try, with links to the upstream vendor documentation. It is deliberately not a copy of the vendor
docs and not a retrieval stack over them: the corpus is small and hand-written, so a transparent
keyword search beats an embedding index that would need reinventing and re-tuning.

## Install and run

```bash
pip install git+https://github.com/littlemex/xprof-knowledge   # or: pip install .
xprof-knowledge-mcp          # serves streamable-http on MCP_PORT (default 8080)
```

Then register `http://127.0.0.1:8080/mcp` as a streamable-http MCP in your client (Claude Code,
Claude Desktop, or any MCP-capable agent). Nothing else is required.

## Tools and resources

- `list_topics(chip="")` — the available playbooks (`gpu` | `neuron` | `common`), id + title + tags.
- `get_topic(topic_id)` — one playbook in full (body + reference links), e.g. `gpu/roofline`.
- `search_knowledge(query, chip="", limit=5)` — the playbooks most relevant to a symptom,
  keyword-ranked, e.g. `search_knowledge("memory bound but occupancy high", "gpu")`.
- Resources: `knowledge://index` (a browsable list) and `knowledge://{chip}/{slug}` (each playbook).
- Prompt `diagnose(chip, symptom)` — primes an agent to search → read → propose one change.

## Content

```
knowledge_mcp/playbooks/
  common/   methodology, metrics glossary
  gpu/      roofline, memory-and-fusion, tensor-cores-and-occupancy
  neuron/   utilization (MFU/MBU/HFU), dma-and-collectives, compile-and-capture
```

Each file is Markdown with a small front matter (`id`, `chip`, `title`, `tags`, `refs`), shipped
inside the wheel. **Add a playbook** by dropping a new `.md` under the right chip directory (or
point `KNOWLEDGE_ROOT` at your own set) — no code change; it is picked up at boot and becomes
searchable and browsable. The playbooks are in English; search with English keywords.

## Testing

```bash
pip install -e . pytest
python -m pytest knowledge_mcp/ -q
```

Tests cover front-matter parsing, list/get/search, duplicate-id detection, and that every packaged
playbook loads and is well-formed.

## Running in a container

`Dockerfile` is a reference image that simply `pip install`s the package and runs the console
script — a sample for containerized or clustered deployments. It assumes no orchestrator; a
Kubernetes deployment that hosts this alongside other MCPs is a separate concern (see the
`distributed-ai` deployment repo), not a dependency of this project.
