"""Shared corpus constants (which files are indexed, which directories are tool-managed)."""

from __future__ import annotations

# Markdown or UTF-8 plain text only; `lens calibrate` converts everything else.
CORPUS_SUFFIXES = frozenset({".md", ".markdown", ".txt"})

# Directories managed by the tool itself: never indexed, never validated, never converted.
#   .lens-cache  vector cache (SQLite)
#   .originals   source PDFs/Office files moved aside by `lens calibrate` (originals: move)
SKIP_CORPUS_DIRS = frozenset({".lens-cache", ".originals"})

# Formats `lens calibrate` can convert to Markdown.
CONVERTIBLE_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".xlsx"})
