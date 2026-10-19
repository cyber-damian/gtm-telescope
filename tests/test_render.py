"""Tests for context building, citation post-processing and confidence in gtm_telescope.render."""

from __future__ import annotations

import json

from gtm_telescope import render
from gtm_telescope.retrieval import Hit

THRESH = 0.38
LOW = 0.02


def mk_hit(
    path: str = "a.md",
    line: int = 1,
    end: int = 5,
    *,
    heading: str | None = "Auth > SSO",
    text: str = "SAML SSO is supported.",
    rrf: float = 0.03,
    cos: float | None = 0.5,
    source: str = "semantic",
    doc_date: str | None = "2026-01-01",
    stale: bool = False,
) -> Hit:
    return Hit(
        path=path,
        line_no=line,
        end_line=end,
        heading=heading,
        text=text,
        rrf_score=rrf,
        cos=cos,
        bm25=None,
        source=source,
        doc_date=doc_date,
        doc_type=None,
        stale=stale,
    )


def process(raw: str, hits: list[Hit], top: Hit | None = None) -> render.RenderResult:
    return render.postprocess_answer(
        raw,
        hits,
        top_hit=top if top is not None else hits[0],
        semantic_threshold=THRESH,
        batch_low_rrf=LOW,
    )


def test_context_header_format_and_full_text() -> None:
    h = mk_hit("product/auth.md", 10, 42, text="full chunk body\nline two", stale=True)
    ctx, used = render.build_context([h], 8)
    assert used == [h]
    first = ctx.splitlines()[0]
    assert first == "[1] product/auth.md:10-42 · Auth › SSO · updated 2026-01-01 (stale)"
    assert "full chunk body\nline two" in ctx


def test_context_respects_max_chunks_and_numbers_from_one() -> None:
    hits = [mk_hit(f"{i}.md") for i in range(5)]
    ctx, used = render.build_context(hits, 3)
    assert len(used) == 3
    assert "[3] 2.md" in ctx and "[4]" not in ctx


def test_sanitize_drops_out_of_range_and_tracks_order() -> None:
    text, used = render.sanitize_citations("A [2]. B [9]. C [1][2]. D [0].", 3)
    assert text == "A [2]. B. C [1][2]. D."
    assert used == [2, 1]


def test_sanitize_list_form() -> None:
    text, used = render.sanitize_citations("Fact [1, 7].", 3)
    assert text == "Fact [1]."
    assert used == [1]


def test_cited_answer_gets_sources_block() -> None:
    hits = [mk_hit("a.md", 3, 9), mk_hit("b.md", 4, 4)]
    res = process("It supports SAML [2]. Also SCIM [1].", hits)
    assert not res.uncited and not res.not_covered
    assert [c.n for c in res.citations] == [2, 1]
    assert res.sources_block == "Sources:\n[2] b.md:4\n[1] a.md:3-9"
    assert res.full_text().endswith("[1] a.md:3-9\n")


def test_uncited_answer_is_flagged() -> None:
    res = process("It supports everything.", [mk_hit()])
    assert res.uncited
    assert res.answer.startswith(render.UNCITED_WARNING)
    assert res.confidence == "low"


def test_all_invalid_citations_count_as_uncited() -> None:
    res = process("It supports everything [5].", [mk_hit()])
    assert res.uncited and "[5]" not in res.answer


def test_not_covered_is_not_flagged_uncited() -> None:
    res = process("Not covered in the docs: FedRAMP status.", [mk_hit()])
    assert res.not_covered and not res.uncited
    assert res.confidence == "not_covered"
    assert render.UNCITED_WARNING not in res.answer
    assert res.sources_block == ""


def test_stale_note_added_when_model_forgot() -> None:
    hits = [mk_hit("old.md", doc_date="2023-01-10", stale=True)]
    res = process("Price is $10 [1].", hits)
    assert "Stale source [1] (verify: source dated 2023-01-10)" in res.answer
    assert res.citations[0].stale


def test_stale_note_not_duplicated_when_model_complied() -> None:
    hits = [mk_hit("old.md", doc_date="2023-01-10", stale=True)]
    res = process("Price is $10 [1] (verify: source dated 2023-01-10).", hits)
    assert res.answer.count("2023-01-10") == 1


def test_confidence_rules() -> None:
    dual = mk_hit(source="dual", rrf=0.05)
    other = mk_hit("b.md")
    assert process("A [1]. B [2].", [dual, other]).confidence == "high"
    # one distinct citation is never high
    assert process("A [1]. B [1].", [dual, other]).confidence == "medium"
    # strong cosine alone also qualifies for high
    strong = mk_hit(cos=THRESH + 0.2)
    assert process("A [1]. B [2].", [strong, other]).confidence == "high"
    # weak top score -> low
    weak = mk_hit(rrf=0.01, cos=0.4)
    assert process("A [1].", [weak]).confidence == "low"
    assert process("A [1]. B [2].", [weak, other]).confidence == "low"
    # middling
    assert process("A [1]. B [2].", [mk_hit(cos=0.45), other]).confidence == "medium"


def test_render_from_hits_no_hits_skips_llm() -> None:
    def boom(_p: str) -> str:
        raise AssertionError("LLM must not be called without hits")

    res = render.render_from_hits(
        "q", [], chat=boom, max_chunks=8, semantic_threshold=THRESH, batch_low_rrf=LOW
    )
    assert res.not_covered and res.confidence == "not_covered"
    assert res.answer == render.NO_HITS_ANSWER


def test_render_from_hits_passes_prompt_and_postprocesses() -> None:
    seen: list[str] = []

    def chat(prompt: str) -> str:
        seen.append(prompt)
        return "SAML is supported [1]."

    hits = [mk_hit("a.md", 1, 5, text="UNIQUE-CHUNK-BODY")]
    res = render.render_from_hits(
        "does it do SAML?",
        hits,
        chat=chat,
        max_chunks=8,
        semantic_threshold=THRESH,
        batch_low_rrf=LOW,
    )
    assert "does it do SAML?" in seen[0] and "UNIQUE-CHUNK-BODY" in seen[0]
    assert res.citations[0].location == "a.md:1-5"


def test_strip_citation_markers() -> None:
    assert render.strip_citation_markers("Yes [1]. Also [2][3].") == "Yes. Also."


def test_format_sales_display_structure() -> None:
    hits = [mk_hit("a.md", 3, 9), mk_hit("b.md", 4, 4)]
    res = process("We support SAML [1]. SCIM too [2].", hits)
    text = render.format_sales_display(
        res, question="Do you support SSO?", mode="short"
    )
    assert "CUSTOMER QUESTION" in text
    assert "Do you support SSO?" in text
    assert "SAY THIS · Quick reply" in text
    assert "MEDIUM" in text
    assert "[1]" not in text.split("IF THEY ASK")[0]
    assert "We support SAML. SCIM too." in text
    assert "IF THEY ASK WHERE THIS IS FROM" in text
    assert "a.md:3-9" in text


def test_format_sales_display_not_covered_shows_check_section() -> None:
    res = process("Not covered in the docs: FedRAMP.", [mk_hit()])
    text = render.format_sales_display(res, question="FedRAMP?", mode="explain")
    assert "NOT IN DOCS" in text
    assert "CHECK BEFORE YOU SEND" in text


def test_json_shape() -> None:
    res = process("A [1].", [mk_hit("a.md", 2, 6)])
    res.elapsed_ms = 12.345
    d = json.loads(res.to_json())
    assert set(d) == {"answer", "citations", "confidence", "elapsed_ms"}
    assert d["citations"][0] == {
        "n": 1,
        "path": "a.md",
        "line": 2,
        "end_line": 6,
        "doc_date": "2026-01-01",
        "stale": False,
    }
    assert d["elapsed_ms"] == 12.3
