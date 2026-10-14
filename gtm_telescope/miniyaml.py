"""Tiny YAML subset parser (no dependencies).

Supports exactly what GTM Telescope needs for frontmatter and ``eval/questions.yaml``:

* ``key: value`` mappings with scalar or inline-list (``[a, "b c"]``) values
* top-level lists of such mappings (``- key: value`` followed by indented ``key: value``)
* full-line comments and trailing `` # comments`` outside quotes

Anything fancier (nesting, block scalars, anchors) is intentionally unsupported.
"""

from __future__ import annotations

from typing import Any

Scalar = str
Value = Any  # str | list[str]


def _strip_comment(s: str) -> str:
    """Drop a trailing `` # comment`` that is not inside quotes."""
    quote: str | None = None
    for i, ch in enumerate(s):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or s[i - 1] in (" ", "\t")):
            return s[:i].rstrip()
    return s.rstrip()


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _split_inline_list(inner: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch == ",":
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    items.append("".join(buf))
    out = [_unquote(i) for i in items]
    return [i for i in out if i != ""]


def parse_value(raw: str) -> Value:
    """Parse a scalar (quotes stripped) or an inline list into ``list[str]``."""
    s = _strip_comment(raw.strip()).strip()
    if s.startswith("[") and s.endswith("]"):
        return _split_inline_list(s[1:-1])
    return _unquote(s)


def _split_key_value(stripped: str) -> tuple[str, str] | None:
    if ":" not in stripped:
        return None
    key, val = stripped.split(":", 1)
    key = key.strip()
    if not key or any(c in key for c in " \t[]{}\"'"):
        return None
    return key, val


def parse_mapping(text: str) -> dict[str, Value]:
    """Parse flat ``key: value`` lines (comments and blank lines ignored)."""
    out: dict[str, Value] = {}
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kv = _split_key_value(stripped)
        if kv is None:
            continue
        out[kv[0]] = parse_value(kv[1])
    return out


def parse_list_of_mappings(text: str) -> list[dict[str, Value]]:
    """Parse a top-level YAML list whose items are flat mappings."""
    items: list[dict[str, Value]] = []
    current: dict[str, Value] | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") or stripped == "-":
            current = {}
            items.append(current)
            stripped = stripped[1:].strip()
            if not stripped:
                continue
        if current is None:
            continue
        kv = _split_key_value(stripped)
        if kv is None:
            continue
        current[kv[0]] = parse_value(kv[1])
    return items


__all__ = ["parse_list_of_mappings", "parse_mapping", "parse_value"]
