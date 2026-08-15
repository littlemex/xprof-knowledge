"""Load and query the profiling-knowledge playbooks.

The knowledge is plain Markdown files with a small YAML front matter (`id`, `chip`, `title`,
`tags`, `refs`), curated in this repo — distilled heuristics plus links to the upstream vendor
docs, never a copy of them. This module reads them once and offers list / get / search. Search is
deliberately a simple keyword score, not an embedding index: the corpus is small and curated, so a
transparent ranking beats a heavyweight retrieval stack we would have to reinvent and maintain.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CHIPS = ("gpu", "neuron", "common")


@dataclass(frozen=True)
class Topic:
    id: str                     # e.g. "gpu/roofline"
    chip: str                   # gpu | neuron | common
    title: str
    tags: tuple[str, ...]
    refs: tuple[str, ...]
    body: str

    def snippet(self, n: int = 240) -> str:
        text = " ".join(self.body.split())
        return text[:n] + ("…" if len(text) > n else "")


_FRONT = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def parse_topic(text: str, fallback_id: str) -> Topic:
    """Parse one playbook file. Front matter is expected but tolerated: missing keys (or no front
    matter at all) fall back to sane defaults so a malformed file surfaces as a thin topic rather
    than crashing the whole server at boot."""
    m = _FRONT.match(text)
    meta: dict = {}
    body = text
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2).strip()
    chip = str(meta.get("chip", "common"))
    return Topic(
        id=str(meta.get("id", fallback_id)),
        chip=chip if chip in CHIPS else "common",
        title=str(meta.get("title", fallback_id)),
        tags=tuple(str(t) for t in (meta.get("tags") or [])),
        refs=tuple(str(r) for r in (meta.get("refs") or [])),
        body=body,
    )


class KnowledgeLibrary:
    def __init__(self, root: str):
        self._root = Path(root)
        self._topics: dict[str, Topic] = {}
        self._load()

    def _load(self) -> None:
        if not self._root.is_dir():
            raise FileNotFoundError(f"knowledge root not found: {self._root}")
        for path in sorted(self._root.rglob("*.md")):
            rel = path.relative_to(self._root).with_suffix("").as_posix()  # e.g. gpu/roofline
            topic = parse_topic(path.read_text(encoding="utf-8"), fallback_id=rel)
            if topic.id in self._topics:
                raise ValueError(f"duplicate topic id {topic.id!r} (front-matter id collides "
                                 f"with another playbook) — ids must be unique")
            self._topics[topic.id] = topic
        if not self._topics:
            raise FileNotFoundError(f"no .md playbooks under {self._root}")

    def list_topics(self, chip: str | None = None) -> list[Topic]:
        ts = [t for t in self._topics.values() if not chip or t.chip == chip]
        return sorted(ts, key=lambda t: (t.chip, t.id))

    def get(self, topic_id: str) -> Topic:
        if topic_id not in self._topics:
            raise KeyError(f"unknown topic {topic_id!r}; see list_topics()")
        return self._topics[topic_id]

    def search(self, query: str, chip: str | None = None, limit: int = 5) -> list[tuple[Topic, int]]:
        """Rank topics by keyword overlap: a title/tag hit weighs more than a body hit. Returns
        (topic, score) for the best matches with a non-zero score. Tokenizes Unicode word runs, so a
        query is never silently emptied by non-ASCII characters — but the playbooks are English, so
        match on English keywords (roofline, coalescing, dma, collective, occupancy, …)."""
        terms = [w for w in re.findall(r"\w+", query.lower(), re.UNICODE) if len(w) > 1]
        scored: list[tuple[Topic, int]] = []
        for t in self.list_topics(chip):
            title = t.title.lower()
            tags = " ".join(t.tags).lower()
            body = t.body.lower()
            score = 0
            for w in terms:
                score += 5 * title.count(w) + 3 * tags.count(w) + min(body.count(w), 3)
            if score:
                scored.append((t, score))
        scored.sort(key=lambda ts: ts[1], reverse=True)
        return scored[:limit]


def default_root() -> str:
    """The playbooks directory shipped inside the package (so it works from a wheel, with no repo
    checkout). Override with KNOWLEDGE_ROOT to point at your own playbook set."""
    env = os.environ.get("KNOWLEDGE_ROOT")
    if env:
        return env
    from importlib.resources import files
    return str(files("knowledge_mcp") / "playbooks")
