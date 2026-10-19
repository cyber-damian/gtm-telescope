# GTM Telescope - Bring the Right Answers into Focus.

#### 🛰️  Telescope

A local multi-retrieval engine that fuses rapid lexical matching with deep semantic understanding. Designed specifically for Go-To-Market teams, it enables individuals to instantly find answers to complex customer questions, dig through product capabilities, and generate high-fidelity technical responses without leaving their workflow.

#### 🔍  How it Works

Most search tools either look for exact words (and miss the context) or look for "vibes" (and miss the specific technical port or spec). Telescope uses a hybrid approach:

* A lexical search (SQLite FTS5, BM25) matches the significant words and any `"quoted phrases"` in your question against Markdown sections, so exact terms like `SCIM`, `TLS 1.3` or `8089` are found even inside a long natural-language question.

* A semantic search is done against the same query to understand the actual intent of your query.

The two ranked lists are fused **per chunk** with Reciprocal Rank Fusion, so a passage found by both ranks highest (`dual`). Each result is a heading-aware chunk of a document with its line range and date; results older than `retrieval.stale_after_days` are flagged `⚠ stale` (never hidden).

#### 🛠  The CLI

There are two distinct ways you can use telescope:

`lens focus <query>`: Instantly resolves the most relevant technical snippets, file locations, and line numbers.

`lens render <query>`: Generate a **cited** answer grounded entirely in your "focused" data. Every claim carries `[n]` markers and a `Sources:` block maps them to `path:line-end_line`. If the docs do not cover the question, the answer says `Not covered in the docs: ...`. Choose the style with `--short` (default), `--story` or `--explain`. It uses Claude, OpenAI or Mistral, depending on which API key you put in `.env`.

**Note:** run **`lens calibrate`** after adding or changing documents (it converts PDF and supported Office files to Markdown and refreshes the vector index). Then use **`lens focus`** / **`lens render`**.

## Requirements

- **macOS on Apple Silicon** - uses Apple's native Silicon-optimized MLX engine.
- **Python 3.10+** - installed or managed by [uv](https://docs.astral.sh/uv/getting-started/installation/).
- **uv** - required for setup and running `lens` (via `bin/lens` → `uv run lens`).

## Quick start

```bash
# Setup the project
git clone https://github.com/cyber-damian/gtm-telescope.git && cd gtm-telescope
./scripts/bootstrap.sh   # uv sync --extra calibrate, ripgrep, optional model prefetch
export PATH="$PWD/bin:$PATH"
cp .env.example .env

# Build the index and start searching
lens start
lens calibrate
lens focus "your question here"
```

Development and tests use the same environment:

```bash
uv sync --extra calibrate --extra dev --extra batch
uv run pytest
uv run ruff check .
```

## Configuration

All tunables and paths are read from [`config.yaml`](config.yaml). Secrets for **`lens render`** are loaded from **`.env`**.

| Block | Purpose |
|-------|---------|
| `llm` | `anthropic_model`, `openai_model`, `mistral_model` (model used for each provider), `max_tokens` (answer length cap), optional `env_file` |
| `storage` | `data_dir` for corpus files |
| `embedding` | `server_url`, `model`, optional `truncate_dim`, timeouts, `max_file_bytes`, optional `allow_remote` |
| `retrieval` | Semantic `top_k`, `semantic_threshold`, `semantic_granularity`, `max_chunk_rows`, hybrid report layout, chunking, lexical and fusion tuning (table below) |
| `calibrate` | `originals`: what happens to PDF/Office files after conversion (`move` \| `keep` \| `delete`) |
| `process` | `subprocess_timeout`, `http_timeout` |

#### `retrieval` keys

| Key | Default | Meaning |
|-----|---------|---------|
| `semantic_granularity` | `markdown` | `markdown` = heading-aware chunks; `line`, `paragraph`, `sentence`, `token` use jina-grep splitting |
| `chunk_tokens` | `350` | Target chunk size (estimated tokens = characters / 3); larger sections are split at line boundaries |
| `min_chunk_tokens` | `40` | Smaller sections merge with the following section under the same parent heading |
| `chunk_overlap_lines` | `2` | Lines repeated between consecutive pieces of a split section |
| `semantic_candidate_top_k` | `15` | Semantic candidates per query |
| `semantic_threshold` | `0.38` | Minimum cosine similarity for a semantic candidate |
| `lexical_top_k` | `15` | Lexical (FTS5 BM25) candidates per query |
| `lexical_max_lines` | `500` | Only used by the ripgrep fallback (when SQLite lacks FTS5) |
| `rrf_k` | `60` | Reciprocal Rank Fusion constant: `score = sum 1 / (rrf_k + rank)` |
| `exact_boost` | `0.02` | Added when a chunk contains every quoted phrase (or every term of a query with 3 terms or fewer) |
| `card_boost` | `0.01` | Added to chunks whose frontmatter `type` is `card` (no effect until cards exist) |
| `stale_after_days` | `365` | Hits whose document date is older are flagged `⚠ stale` (not demoted) |
| `render_max_chunks` | `8` | `lens render`: number of top chunks (full text) sent to the LLM |
| `batch_low_rrf` | `0.02` | `lens render`: confidence is `low` when the top fused score is below this |
| `hybrid_report_max_rows` | `10` | Maximum results |

Changing the chunking keys or `semantic_granularity` re-indexes the corpus on the next `lens calibrate`.

#### Document metadata (frontmatter)

Markdown files may start with a frontmatter block. It is stripped from the indexed text and used as metadata:

```yaml
---
title: Platform pricing and limits
type: battlecard          # free-form label, stored lower-case
updated: 2023-01-10       # document date; also accepted: date
---
```

The date shown for a hit is `updated`/`date`, else `source_mtime` (written by `lens calibrate`), else the file's modification time.

## Data Storage 

All data and documents should be put in **`data-storage/`** by default. You can change the default location in [`config.yaml`](config.yaml). There are several examples files provided to give you a feel how telescope works (remove them when you are finished learning).

**`data-storage/snyk/`** is gitignored: use it for local-only corpus (for example Snyk-related docs). After adding files there, run **`lens calibrate`** so they are indexed.

### Calibrate

`lens calibrate` acts as a micro ETL pipeline. It will discover **PDF** and supported **Office OOXML** files (`.docx`, `.pptx`, `.xlsx`) under the data directory and converts them to Markdown. What happens to the original is controlled by `calibrate.originals` in [`config.yaml`](config.yaml):

| `originals` | Behavior |
|-------------|----------|
| `move` (default) | After a successful conversion the original is moved to `<data_dir>/.originals/<same relative path>` (never overwriting an earlier original). Nothing is lost if conversion quality is poor. |
| `keep` | The original stays next to the generated `.md`. Re-runs skip files whose `.md` is already up to date, so hand edits to the Markdown are not overwritten. |
| `delete` | The original is removed (the pre-v2 behavior). |

Each generated Markdown file starts with provenance frontmatter (`source`, `source_mtime`, `converted`). `.originals/` and `.lens-cache/` are never indexed or converted. Calibrate then refreshes the index (SQLite vector + FTS5 cache under `.lens-cache`). `lens focus`, `lens render` and `lens eval` never index; they only read the cache and ask you to run `lens calibrate` when files changed.

Upgrading from v1: the first `lens calibrate` rebuilds the cache (schema v2) and prints a one-line notice.

### Example Searches

The repo ships a small **sample dataset** under [`data-storage/`](data-storage/). You can run the following examples to see how telescope works. Try:

```bash
# Operational / incident content
lens focus "how should we handle a database outage during peak traffic"

# Cooking / recipe content 
lens focus "vegetarian dinner with legumes and warm spices"

# Marker string (also participates in semantic retrieval)
lens focus 'GTM_TELESCOPE_LEXICAL_ONLY_MARKER_7f4c2b91'

# Exact terms inside a natural-language question
lens focus 'Do you support SCIM provisioning?'
lens focus 'Do you require "TLS 1.3"?'

# Dual detection (found by both lexical and semantic search)
lens focus 'failover to the standby database'
```

Output is a ranked table followed by the top snippets:

```text
Rank | File                           | Lines | Source | RRF    | Cos   | Date
-----+--------------------------------+-------+--------+--------+-------+-----------
1    | operations/incident-runbook.md | 1-9   | dual   | 0.0528 | 0.545 | 2026-10-14
...
[1] operations/incident-runbook.md:1-9 · Incident response runbook › Database outage during peak traffic
2. **Failover**: if the primary writer is unhealthy, promote the warm standby and ...
```

`Source` is `dual` (both searches), `lexical` or `semantic`; stale documents show `⚠ stale` next to their date. Paths are relative to the data directory.

### Evaluating retrieval

`lens eval` measures retrieval quality against a golden set so tuning (thresholds, chunking, models) is not guesswork. Each entry in [`eval/questions.yaml`](eval/questions.yaml) is:

```yaml
- q: "Do you support SCIM provisioning?"
  expect: ["product/feature-spec-auth.md"]   # corpus-relative; any listed file counts
  tags: [exact-term, live]
```

It prints the rank of the first expected file per question (or `miss`), then **Recall@k**, **MRR** and per-tag breakdowns. Questions with `expect: []` are "should not match" cases: they are excluded from recall and reported as an **abstain rate** (share that returned no hits). The command always exits 0; it is a measurement, not a gate. Run `lens calibrate` first.

### Commands

| Command | Purpose |
|--------|---------|
| `lens start` | Start jina-grep embedding server (pass-through args supported) |
| `lens stop` | Stop embedding server |
| `lens status` | Show embedding server status + configured URL/port |
| `lens calibrate` | Convert PDF/Office → Markdown; refresh vector cache |
| `lens focus <text>` | Hybrid search engine |
| `lens render [--short\|--story\|--explain] [--json] <text>` | Cited LLM answer (`[n]` markers + `Sources:` block) in the chosen style (default `--short`); `--json` prints `{answer, citations, confidence, elapsed_ms}` |
| `lens render [--short\|--story\|--explain] --batch FILE [--question-col NAME] [--out FILE] [--sheet NAME]` | Answer every row of a CSV/XLSX questionnaire (see below) |
| `lens eval [--file F] [--k 3,10] [--json]` | Run golden questions ([`eval/questions.yaml`](eval/questions.yaml)) through the `focus` retrieval path; prints Recall@k and MRR |
| `lens cache info` | Vector cache statistics |
| `lens cache clear` | Remove vector cache on disk |

### `lens render`: providers

Copy [`.env.example`](.env.example) to `.env` and fill in one key. The provider is chosen from the keys that are set, in this order:

| Key in `.env` | Provider | Model setting in `config.yaml` |
|---------------|----------|--------------------------------|
| `ANTHROPIC_API_KEY` | Claude | `llm.anthropic_model` |
| `OPENAI_API_KEY` | OpenAI | `llm.openai_model` |
| `MISTRAL_API_KEY` | Mistral | `llm.mistral_model` |

If several keys are set, the first one in the table wins. To pick a different one, set `LLM_PROVIDER=anthropic`, `openai` or `mistral`. If no key is set, `lens render` exits with an error.

What is sent to the provider: the question, the top `render_max_chunks` retrieved excerpts (with their file paths and dates), and the prompt template for the chosen style. Nothing else from your corpus is sent.

### `lens render`: answer styles

| Flag | Output |
|------|--------|
| `--short` (default) | Two sentences you can send to a customer as-is |
| `--story` | 6 to 8 sentences built around a customer story found in the retrieved sources. If none of the sources contains a story, the answer still covers the question and ends with `Not covered in the docs: no customer story found`. |
| `--explain` | Up to two detailed paragraphs explaining the concept |

```bash
lens render --short "Do you support SCIM provisioning?"
lens render --story "How do customers handle a database outage?"
lens render --explain 'How does "TLS 1.3" encryption work in the platform?'
```

Each style is a separate prompt template in [`gtm_telescope/prompt_templates/`](gtm_telescope/prompt_templates/): `rules.md` holds the grounding and citation rules shared by every style, and `short.md`, `story.md` and `explain.md` add the format for each one. Edit those files to change the tone or length.

Terminal output is laid out for live calls: **CUSTOMER QUESTION**, **SAY THIS** (citation markers stripped so you can read it aloud), **IF THEY ASK WHERE THIS IS FROM** (doc paths and dates), and **CHECK BEFORE YOU SEND** when confidence is low or sources are stale. Post-processing is deterministic: invalid citations are removed; uncited answers are flagged; stale sources get a verify note. Use `--json` for the raw answer text and citation objects.

### Batch questionnaires

`lens render --batch questions.csv` (or `.xlsx`, which needs `uv sync --extra batch`) answers each row in the chosen style (add `--short`, `--story` or `--explain`) and writes `questions.answers.csv` next to the input (override with `--out`). The question column is `--question-col`, else the first header containing "question", "requirement" or "query", else the first column. Input columns are passed through and four are appended: `answer`, `citations` (`path:lines; ...`), `confidence`, `stale_sources`. Progress goes to stderr, and the output is rewritten after every row, so an interrupted run **resumes** when you re-run the same command (rows whose `answer` is filled are kept).

`confidence` values: `not_covered` (the model said so, or nothing was retrieved); `high` (at least two distinct cited sources and the top hit is `dual`, or has cosine ≥ `semantic_threshold + 0.15`); `low` (no valid citation, or top fused score < `batch_low_rrf`); otherwise `medium`. Review `low` and `not_covered` rows by hand.

## License

**gtm-telescope** (this repository) is licensed under the [MIT License](LICENSE).

This project depends on [**jina-grep**](https://github.com/jina-ai/jina-grep-cli), which is licensed under the **Apache License 2.0**. Upstream terms apply to that dependency.
