"""Query parsing for lexical (FTS5) search.

``parse_query`` turns a natural-language question into quoted phrases and significant terms.
``to_fts5`` builds a MATCH expression in which *every* piece is a double-quoted FTS5 string, so
user text can never be interpreted as FTS5 syntax (``NEAR``, ``AND``/``OR``/``NOT``, ``*``,
``:``, parentheses, ``^`` ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Small built-in English list: question words, articles, auxiliaries, and filler that is
# useless for exact-term matching in sales-engineering questions.
STOPWORDS = frozenset(
    {
        "a", "about", "also", "an", "and", "any", "are", "as", "at", "be", "been", "being",
        "by", "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
        "have", "how", "i", "if", "in", "is", "it", "its", "may", "me", "might", "much",
        "must", "my", "not", "of", "on", "or", "our", "ours", "please", "should", "so",
        "support", "supported", "supports", "tell", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "those", "to", "us", "was", "we", "were", "what",
        "when", "where", "which", "who", "whom", "whose", "why", "will", "with", "would",
        "you", "your", "yours",
    }
)  # fmt: skip

_PHRASE_RE = re.compile(r'"([^"]+)"')
_EDGE_PUNCT = "\"'`,;:!?()[]{}<>*^~|\\/"
_KEEP_CHARS = ("_", "-", ".")


@dataclass(frozen=True)
class ParsedQuery:
    phrases: tuple[str, ...]
    terms: tuple[str, ...]
    raw: str

    @property
    def is_empty(self) -> bool:
        return not self.phrases and not self.terms


def _has_alnum(s: str) -> bool:
    return any(c.isalnum() for c in s)


def _clean_token(tok: str) -> str:
    t = tok.strip(_EDGE_PUNCT)
    # Sentence-final or leading dots are noise ("SCIM.", ".env"); inner dots stay (TLS1.3).
    t = t.rstrip(".").lstrip(".")
    return t


def _is_significant(t: str) -> bool:
    if not _has_alnum(t):
        return False
    if any(c.isdigit() for c in t) or any(c in t for c in _KEEP_CHARS):
        return True  # v2, 8089, TLS1.3, rate-limit, snake_case stay verbatim
    return t not in STOPWORDS and len(t) > 1


def parse_query(q: str) -> ParsedQuery:
    """Split ``q`` into quoted phrases (kept intact) and lowercase significant terms."""
    phrases: list[str] = []
    for m in _PHRASE_RE.finditer(q):
        p = " ".join(m.group(1).split())
        if _has_alnum(p) and p not in phrases:
            phrases.append(p)
    rest = _PHRASE_RE.sub(" ", q).replace('"', " ")
    terms: list[str] = []
    for tok in rest.split():
        t = _clean_token(tok).lower()
        if _is_significant(t) and t not in terms:
            terms.append(t)
    return ParsedQuery(phrases=tuple(phrases), terms=tuple(terms), raw=q)


def _quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def to_fts5(parsed: ParsedQuery) -> str:
    """FTS5 MATCH expression: quoted phrases and quoted terms, OR-ed. ``""`` if nothing to match."""
    pieces = [_quote(p) for p in parsed.phrases] + [_quote(t) for t in parsed.terms]
    return " OR ".join(pieces)


__all__ = ["STOPWORDS", "ParsedQuery", "parse_query", "to_fts5"]
