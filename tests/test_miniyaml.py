"""Tests for gtm_telescope.miniyaml."""

from __future__ import annotations

from gtm_telescope import miniyaml


def test_parse_value_scalars_and_quotes() -> None:
    assert miniyaml.parse_value('"hello world"') == "hello world"
    assert miniyaml.parse_value("'it''s'") == "it''s"
    assert miniyaml.parse_value("plain") == "plain"
    assert miniyaml.parse_value("") == ""


def test_parse_value_inline_list() -> None:
    assert miniyaml.parse_value("[a, b]") == ["a", "b"]
    assert miniyaml.parse_value('["a, with comma", b]') == ["a, with comma", "b"]
    assert miniyaml.parse_value("[]") == []


def test_trailing_comment_stripped_outside_quotes_only() -> None:
    assert miniyaml.parse_value('["x.md"]   # note') == ["x.md"]
    assert miniyaml.parse_value('"a # b"') == "a # b"
    assert miniyaml.parse_value("draft # trailing") == "draft"
    assert miniyaml.parse_value("C#") == "C#"


def test_parse_mapping_skips_comments_and_blank() -> None:
    text = "# header\ntype: battlecard\n\nupdated: 2023-01-10\ntags: [a, b]\n"
    assert miniyaml.parse_mapping(text) == {
        "type": "battlecard",
        "updated": "2023-01-10",
        "tags": ["a", "b"],
    }


def test_parse_mapping_value_may_contain_colon() -> None:
    assert miniyaml.parse_mapping("title: Foo: the bar") == {"title": "Foo: the bar"}


def test_parse_list_of_mappings() -> None:
    text = """
# comment
- q: "Do you support SCIM?"
  expect: ["a.md", "b.md"]   # any match
  tags: [exact-term, live]

- q: 'Say "TLS 1.3"'
  expect: []
"""
    items = miniyaml.parse_list_of_mappings(text)
    assert items == [
        {"q": "Do you support SCIM?", "expect": ["a.md", "b.md"], "tags": ["exact-term", "live"]},
        {"q": 'Say "TLS 1.3"', "expect": []},
    ]
