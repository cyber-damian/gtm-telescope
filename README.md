# GTM Telescope - Bring the Right Answers into Focus.

#### 🛰️  Telescope

A local multi-retrieval engine that fuses rapid lexical matching with deep semantic understanding. Designed specifically for Go-To-Market teams, it enables individuals to instantly find answers to complex customer questions, dig through product capabilities, and generate high-fidelity technical responses without leaving their workflow.

#### 🔍  How it Works

Most search tools either look for exact words (and miss the context) or look for "vibes" (and miss the specific technical port or spec). Telescope uses a hybrid approach:

* A high-speed scan engine looks through thousands of lines of files in microseconds to find exact keyword matches.

* A semantic search is done against the same query to understand the actual intent of your query.

Together this hybrid search approach allows for the best responses to be surfaced and measured against your original query.

#### 🛠  The CLI

There are two distinct ways you can use telescope:

`lens focus <query>`: Instantly resolves the most relevant technical snippets, file locations, and line numbers.

`lens render <query>`: Generate an LLM based response grounded entirely in your "focused" data.

**Note:** run **`lens calibrate`** after adding or changing documents (it converts PDF and supported Office files to Markdown and refreshes the vector index). Then use **`lens focus`** / **`lens render`**.

## Requirements

- **macOS on Apple Silicon** - uses Apple's native Silicon-optimized MLX engine.
- **Python 3.10+** - accessible in your PATH.
- **uv** - (optional but recommended); otherwise the bootstrap script uses `python3 -m venv` and `pip`.

## Quick start

```bash
# Setup the project
git clone https://github.com/cyber-damian/gtm-telescope.git && cd gtm-telescope
./scripts/bootstrap.sh
export PATH="$PWD/bin:$PATH"
cp .env.example .env

# Build the index and start searching
lens start
lens calibrate
lens focus "your question here"
```

## Configuration

All tunables and paths are read from [`config.yaml`](config.yaml). Secrets for **`lens render`** are loaded from **`.env`**.

| Block | Purpose |
|-------|---------|
| `llm` | LLM endpoint, model, `api_key_env` (variable name populated from `.env`), optional `env_file` |
| `storage` | `data_dir` for corpus files |
| `embedding` | `server_url`, `model`, optional `truncate_dim`, timeouts, `max_file_bytes`, optional `allow_remote` |
| `retrieval` | Semantic `top_k`, `semantic_threshold`, `semantic_granularity`, `max_chunk_rows`, hybrid report layout |
| `process` | `subprocess_timeout`, `http_timeout` |

## Data Storage 

All data and documents should be put in **`data-storage/`** by default. You can change the default location in [`config.yaml`](config.yaml). There are several examples files provided to give you a feel how telescope works (remove them when you are finished learning).

### Calibrate

`lens calibrate` acts as a micro ETL pipeline. It will discover **PDF** and supported **Office OOXML** files (`.docx`, `.pptx`, `.xlsx`) under the data directory, converts them to Markdown, and removes the originals on success. It then refreshes the embedding index (SQLite vector cache under `.lens-cache`).

### Example Searches

The repo ships a small **sample dataset** under [`data-storage/`](data-storage/). You can run the following examples to see how telescope works. Try:

```bash
# Operational / incident content
lens focus "how should we handle a database outage during peak traffic"

# Cooking / recipe content 
lens focus "vegetarian dinner with legumes and warm spices"

# Marker string (also participates in semantic retrieval)
lens focus 'GTM_TELESCOPE_LEXICAL_ONLY_MARKER_7f4c2b91'

# Dual detection
lens focus 'warm standby'
```

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
| `lens render <text>` | LLM generative response |
| `lens eval [--file F] [--k 3,10] [--json]` | Run golden questions ([`eval/questions.yaml`](eval/questions.yaml)) through the `focus` retrieval path; prints Recall@k and MRR |
| `lens cache info` | Vector cache statistics |
| `lens cache clear` | Remove vector cache on disk |

For **`render`**, set variables in **`.env`** (see `llm.api_key_env` and `llm.env_file` in [`config.yaml`](config.yaml)).

## License

**gtm-telescope** (this repository) is licensed under the [MIT License](LICENSE).

This project depends on [**jina-grep**](https://github.com/jina-ai/jina-grep-cli), which is licensed under the **Apache License 2.0**. Upstream terms apply to that dependency.
