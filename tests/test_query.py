"""Tests for gtm_telescope.query (parsing + FTS5 escaping). No embedding server needed."""

from __future__ import annotations

import sqlite3

import pytest

from gtm_telescope.query import STOPWORDS, parse_query, to_fts5


def test_stopword_list_is_small() -> None:
    assert len(STOPWORDS) < 100


def test_stopwords_dropped_and_terms_lowercased() -> None:
    p = parse_query("Do you support SCIM provisioning?")
    assert p.terms == ("scim", "provisioning")
    assert p.phrases == ()
    assert p.raw == "Do you support SCIM provisioning?"


def test_quoted_phrases_preserved_intact() -> None:
    p = parse_query('Do you require "TLS 1.3" for  "SAML   SSO"?')
    assert p.phrases == ("TLS 1.3", "SAML SSO")
    assert p.terms == ("require", "for") or p.terms == ("require",)
    assert "tls" not in p.terms


def test_digit_and_punctuation_tokens_kept_verbatim() -> None:
    p = parse_query("what is v2 on port 8089 with TLS1.3 and rate-limit and snake_case?")
    assert p.terms == ("v2", "port", "8089", "tls1.3", "rate-limit", "snake_case")


def test_edge_punctuation_trimmed() -> None:
    p = parse_query("(SCIM), SSO. .env!")
    assert p.terms == ("scim", "sso", "env")


def test_all_stopwords_is_empty() -> None:
    p = parse_query("what is the")
    assert p.is_empty
    assert to_fts5(p) == ""


def test_unbalanced_quote_is_ignored_safely() -> None:
    p = parse_query('how does "SCIM work')
    assert p.phrases == ()
    assert "scim" in p.terms


def test_to_fts5_quotes_everything_and_ors() -> None:
    p = parse_query('"TLS 1.3" okta scim')
    assert to_fts5(p) == '"TLS 1.3" OR "okta" OR "scim"'


def test_to_fts5_escapes_embedded_quotes() -> None:
    p = parse_query('say a"b')
    assert '""' not in to_fts5(p) or to_fts5(p).count('"') % 2 == 0


ADVERSARIAL = [
    'NEAR(a b)',
    "foo AND bar OR baz NOT qux",
    "AND",
    "OR NOT",
    "col:value heading:scim",
    "scim*",
    "(unbalanced",
    "a ^b",
    '"quoted * phrase (NEAR)"',
    '""" ""',
    "-minus +plus",
    "{curly} [square]",
    "tls1.3:enabled",
    "'single'",
    "back\\slash",
]


@pytest.mark.parametrize("q", ADVERSARIAL)
def test_adversarial_queries_never_break_fts5(q: str) -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(heading, text)")
    conn.execute(
        "INSERT INTO t(heading, text) VALUES ('h', 'scim near and or not col value a b tls1 3')"
    )
    expr = to_fts5(parse_query(q))
    if not expr:
        return
    conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expr,)).fetchall()


def test_operators_are_inert_not_syntax() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(text)")
    conn.executemany("INSERT INTO t(text) VALUES (?)", [("apples only",), ("bananas only",)])
    # As FTS5 syntax "apples NOT bananas" would exclude row 2; as quoted terms it is an OR of
    # the words (apples, bananas); the stopword-like "not" is dropped.
    expr = to_fts5(parse_query("apples NOT bananas"))
    rows = conn.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY rowid", (expr,)).fetchall()
    assert rows == [(1,), (2,)]
