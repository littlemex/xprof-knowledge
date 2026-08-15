"""Tests for the knowledge library: front-matter parsing, list/get/search, and that the packaged
playbooks load and are well-formed."""
from __future__ import annotations

import pytest

from .library import CHIPS, KnowledgeLibrary, default_root, parse_topic


def test_parse_topic_front_matter():
    t = parse_topic("---\nid: gpu/x\nchip: gpu\ntitle: X\ntags: [a, b]\n"
                    "refs:\n  - http://e/1\n---\nbody text here", fallback_id="gpu/x")
    assert t.id == "gpu/x" and t.chip == "gpu" and t.title == "X"
    assert t.tags == ("a", "b") and t.refs == ("http://e/1",)
    assert t.body == "body text here"


def test_parse_topic_defaults_are_safe():
    t = parse_topic("no front matter", fallback_id="common/y")
    assert t.id == "common/y" and t.chip == "common" and t.body == "no front matter"


def test_parse_topic_unknown_chip_falls_back_to_common():
    t = parse_topic("---\nchip: tpu\n---\nb", fallback_id="z")
    assert t.chip == "common"


def _lib(tmp_path):
    for rel, body in {
        "gpu/roofline.md": "---\nid: gpu/roofline\nchip: gpu\ntitle: Roofline\ntags: [roofline, memory-bound, occupancy]\n---\nA kernel below the roofline is often memory bound.",
        "neuron/dma.md": "---\nid: neuron/dma\nchip: neuron\ntitle: DMA\ntags: [dma, collective]\n---\nDMA queues stall the engines.",
        "common/method.md": "---\nid: common/method\nchip: common\ntitle: Method\ntags: [loop]\n---\nProfile then analyze then improve.",
    }.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return KnowledgeLibrary(str(tmp_path))


def test_list_and_filter_by_chip(tmp_path):
    lib = _lib(tmp_path)
    assert len(lib.list_topics()) == 3
    assert [t.id for t in lib.list_topics("gpu")] == ["gpu/roofline"]


def test_get_and_unknown(tmp_path):
    lib = _lib(tmp_path)
    assert lib.get("neuron/dma").title == "DMA"
    with pytest.raises(KeyError):
        lib.get("gpu/nope")


def test_search_ranks_title_over_body(tmp_path):
    lib = _lib(tmp_path)
    hits = lib.search("roofline memory bound", chip="gpu")
    assert hits and hits[0][0].id == "gpu/roofline" and hits[0][1] > 0
    # chip scoping excludes other chips
    assert all(t.chip == "gpu" for t, _ in hits)


def test_empty_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeLibrary(str(tmp_path / "does-not-exist"))


# --- the real packaged playbooks -----------------------------------------------------------

def test_packaged_playbooks_load_and_are_wellformed():
    lib = KnowledgeLibrary(default_root())
    topics = lib.list_topics()
    assert len(topics) >= 6
    chips = {t.chip for t in topics}
    assert chips <= set(CHIPS) and {"gpu", "neuron", "common"} <= chips
    for t in topics:
        assert t.title and t.body, f"{t.id} missing title/body"
        assert t.id.startswith(t.chip + "/"), f"{t.id} id/chip mismatch"
