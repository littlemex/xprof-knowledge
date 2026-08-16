# accelprof-knowledge

[![ci](https://github.com/littlemex/accelprof-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/littlemex/accelprof-knowledge/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

**An MCP server that answers "the profile says X — what do I do about it?" for NVIDIA GPU and AWS
Neuron.**

It serves a small set of curated tuning playbooks, each written as symptom → likely cause → what to
check → what to try, with links to the upstream vendor documentation. An agent (or you) driving the
profile → analyze → improve loop searches for a symptom and gets back the relevant playbook.

The corpus is small and hand-written, so search is a transparent keyword rank — not an embedding
index that would need building and tuning.

It needs no external services — no accelerator, no cloud account, no cluster; its only runtime
dependencies are the `mcp` SDK and `PyYAML`.

## Install and run

```bash
pip install accelprof-knowledge
accelprof-knowledge-mcp            # streamable-http on MCP_PORT (default 8080)
```

Register it with any MCP client — for example Claude Code:

```bash
claude mcp add --transport http accelprof-knowledge http://127.0.0.1:8080/mcp
```

Two optional environment variables: `MCP_PORT` sets the listen port (default `8080`), and
`KNOWLEDGE_ROOT` points the server at your own playbook directory instead of the packaged one.

## Tools, resources, and prompt

| Tool | What it does |
|---|---|
| `list_topics(chip="")` | The available playbooks (`gpu` \| `neuron` \| `common`) — id, title, tags. |
| `get_topic(topic_id)` | One playbook in full — body plus reference links, e.g. `gpu/roofline`. |
| `search_knowledge(query, chip="", limit=5)` | The playbooks most relevant to a symptom, keyword-ranked. |

```jsonc
// search_knowledge("memory bound but occupancy is high", chip="gpu")
{ "query": "memory bound but occupancy is high", "chip": "gpu", "count": 2,
  "results": [
    { "id": "gpu/roofline", "chip": "gpu", "score": 12.0,
      "title": "Roofline diagnosis — compute-bound vs memory-bound vs latency-bound",
      "snippet": "The roofline places a kernel by its arithmetic intensity (FLOPs per byte)…" },
    { "id": "gpu/memory-and-fusion", "chip": "gpu", "score": 7.0, "title": "…", "snippet": "…" }
  ] }
```

Resources `knowledge://index` (a browsable list) and `knowledge://{chip}/{slug}` (each playbook)
expose the same content to clients that prefer resources, and the `diagnose(chip, symptom)` prompt
primes an agent to search → read → propose one change.

## Content

```
knowledge_mcp/playbooks/
  common/   methodology · metrics glossary
  gpu/      roofline · memory-and-fusion · tensor-cores-and-occupancy
  neuron/   utilization (MFU/MBU/HFU) · dma-and-collectives · compile-and-capture
```

Each file is Markdown with a small front matter (`id`, `chip`, `title`, `tags`, `refs`) and is
shipped inside the wheel. Add a playbook by dropping a new `.md` under the right chip directory, with
no code change; it is loaded at startup and becomes searchable and browsable. The playbooks are in
English; search with English keywords.

## Testing

```bash
pip install -e ".[test]"
python -m pytest knowledge_mcp/ -q
```

The suite covers front-matter parsing, list/get/search, duplicate-id detection, and that every
packaged playbook loads and is well-formed.

## Hosting

`Dockerfile` is a reference image that installs the package and runs the console script. It assumes
no orchestrator; hosting this MCP alongside others is a separate concern handled by a deployment repo
(`distributed-ai`), not a dependency of this one.

## Related projects

- **[accelprof](https://github.com/littlemex/accelprof)** — an experiment store plus an analysis MCP
  that maps a run to its profile files and analyzes them. Pair it with this one so a finding leads to
  a next step. When hosting both, give each its own `MCP_PORT`.
- The official **MLflow MCP** — run discovery and search.

## Contributing & license

Playbooks are the main contribution surface: add a Markdown file under
`knowledge_mcp/playbooks/<chip>/` with the front matter above, keep it distilled (symptom → cause →
what to check → what to try), and link the upstream source rather than copying it. Code changes
should keep `python -m pytest knowledge_mcp/ -q` green.

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
