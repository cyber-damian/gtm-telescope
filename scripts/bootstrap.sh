#!/usr/bin/env bash
# Initial setup and health checks.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
cd "$ROOT"

CHECK_ONLY=0
FORCE_INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --check-only) CHECK_ONLY=1 ;;
    --force-install) FORCE_INSTALL=1 ;;
    *)
      echo "error: unknown option: $arg" >&2
      echo "usage: $0 [--check-only | --force-install]" >&2
      exit 2
      ;;
  esac
done

echo "Project root: $ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found on PATH" >&2
  exit 1
fi

if ! python3 -c 'import sys; assert sys.version_info >= (3, 10), "need 3.10+"' 2>/dev/null; then
  echo "error: Python 3.10+ required (jina-grep). Found:" >&2
  python3 --version >&2
  exit 1
fi
echo "Python OK: $(python3 --version)"

# Bundled ripgrep → resources/rg (pinned release)
RG_VER="14.1.1"
RES_RG="$ROOT/resources/rg"
# SHA256 checksums for the upstream tarballs from
# https://github.com/BurntSushi/ripgrep/releases/tag/14.1.1
RG_SHA256_aarch64_apple_darwin="24ad76777745fbff131c8fbc466742b011f925bfa4fffa2ded6def23b5b937be"
RG_SHA256_x86_64_apple_darwin="fc87e78f7cb3fea12d69072e7ef3b21509754717b746368fd40d88963630e2b3"
ensure_bundled_rg() {
  if [[ -x "$RES_RG" ]]; then
    echo "bundled ripgrep OK: $RES_RG"
    return 0
  fi
  if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "error: bundled ripgrep missing or not executable: $RES_RG" >&2
    echo "  Run ./scripts/bootstrap.sh without --check-only (needs network once) to download it." >&2
    return 1
  fi
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: automatic ripgrep download is only implemented for macOS (Darwin)." >&2
    echo "  Install rg on PATH or place an executable at $RES_RG" >&2
    return 1
  fi
  local arch rga tmp tgz url expected actual
  arch="$(uname -m)"
  case "$arch" in
    arm64)
      rga="aarch64-apple-darwin"
      expected="$RG_SHA256_aarch64_apple_darwin"
      ;;
    x86_64)
      rga="x86_64-apple-darwin"
      expected="$RG_SHA256_x86_64_apple_darwin"
      ;;
    *)
      echo "error: unsupported uname -m for bundled ripgrep: $arch" >&2
      return 1
      ;;
  esac
  tgz="ripgrep-${RG_VER}-${rga}.tar.gz"
  url="https://github.com/BurntSushi/ripgrep/releases/download/${RG_VER}/${tgz}"
  echo "Downloading ripgrep ${RG_VER} (${rga}) to resources/rg ..." >&2
  mkdir -p "$ROOT/resources"
  tmp="$(mktemp -d)"
  cleanup() { rm -rf "$tmp"; }
  trap cleanup EXIT
  curl -fsSL --connect-timeout 10 --max-time 120 --retry 3 --retry-delay 2 \
    "$url" -o "$tmp/$tgz"
  if ! command -v shasum >/dev/null 2>&1; then
    echo "error: shasum not found; cannot verify ripgrep download integrity" >&2
    return 1
  fi
  actual="$(shasum -a 256 "$tmp/$tgz" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "error: ripgrep tarball SHA256 mismatch for $tgz" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
  echo "ripgrep tarball SHA256 OK: $actual"
  tar -xzf "$tmp/$tgz" -C "$tmp"
  mv "$tmp/ripgrep-${RG_VER}-${rga}/rg" "$RES_RG"
  chmod +x "$RES_RG"
  trap - EXIT
  cleanup
  echo "bundled ripgrep OK: $RES_RG"
}
ensure_bundled_rg

if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "error: pyproject.toml missing in $ROOT" >&2
  exit 1
fi

VENV_PY="$ROOT/.venv/bin/python"
VENV_PIP="$ROOT/.venv/bin/pip"
JINA_VENV="$ROOT/.venv/bin/jina-grep"

run_install() {
  if command -v uv >/dev/null 2>&1; then
    echo "Using uv to create/update .venv and install dependencies..."
    uv venv --python python3
    # Pin installs to this repo's .venv. Without --python, uv pip can target
    # another environment and skip optional extras (e.g. docling for calibrate).
    uv pip install --python "$ROOT/.venv/bin/python" ".[calibrate]"
  else
    echo "uv not found; using python3 -m venv and pip..."
    python3 -m venv .venv
    "$VENV_PY" -m pip install -U pip
    "$VENV_PIP" install ".[calibrate]"
  fi
  echo "Note: jina-grep targets Apple Silicon (MLX on macOS); other platforms may not work."
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  :
elif [[ "$FORCE_INSTALL" -eq 1 ]]; then
  echo "(--force-install) Re-running install..."
  run_install
elif [[ ! -x "$JINA_VENV" ]]; then
  echo "Project venv missing or jina-grep not installed; bootstrapping..."
  run_install
else
  echo "Skipping install: $JINA_VENV already present (use --force-install to reinstall)"
fi

YAML_PY="python3"
if [[ -x "$VENV_PY" ]]; then
  YAML_PY="$VENV_PY"
fi
DATA_DIR="$(ROOT="$ROOT" PYTHONPATH="$ROOT" "$YAML_PY" <<'PY'
import os
from pathlib import Path

from gtm_telescope.config import parse_yaml_block

root = Path(os.environ["ROOT"])
cfg_path = root / "config.yaml"
raw = "data-storage"
if cfg_path.is_file():
    storage = parse_yaml_block(cfg_path.read_text(encoding="utf-8"), "storage")
    raw = (storage.get("data_dir") or "data-storage").strip()
p = Path(raw)
resolved = p if p.is_absolute() else (root / p).resolve()
print(resolved)
PY
)"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Creating data directory at $DATA_DIR"
  mkdir -p "$DATA_DIR"
fi
echo "data storage OK: $DATA_DIR"

PREFETCH_MARKER="$DATA_DIR/.gtm_telescope_bootstrap_prefetch_ok"

LENS_CLI="$ROOT/bin/lens"
if [[ ! -f "$LENS_CLI" ]]; then
  echo "warning: bin/lens not found at $LENS_CLI" >&2
else
  echo "bin/lens OK: $LENS_CLI"
fi

JINA_GREP=""
if [[ -x "$JINA_VENV" ]]; then
  JINA_GREP="$JINA_VENV"
elif command -v jina-grep >/dev/null 2>&1; then
  JINA_GREP="$(command -v jina-grep)"
fi

if [[ -z "$JINA_GREP" ]]; then
  echo "error: jina-grep not found (no $JINA_VENV and not on PATH)" >&2
  if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "  Run without --check-only to install, or: ./scripts/bootstrap.sh" >&2
  else
    echo "  Re-run: ./scripts/bootstrap.sh --force-install" >&2
  fi
  exit 1
fi
echo "jina-grep OK: $JINA_GREP"

if [[ -f "$PREFETCH_MARKER" && "${FORCE_PREFETCH:-}" != "1" ]]; then
  echo "Skipping MLX + Docling prefetch (marker $PREFETCH_MARKER exists; set FORCE_PREFETCH=1 to re-run)"
else
  PREFETCH_RAN=0
  if [[ "${SKIP_MODEL_PREFETCH:-}" == "1" ]]; then
    echo "Skipping MLX nano model prefetch (SKIP_MODEL_PREFETCH=1)"
  elif [[ -x "$VENV_PY" ]]; then
    echo "Prefetching default nano MLX weights (jinaai/jina-embeddings-v5-text-nano-mlx)..." >&2
    echo "  This can take a few minutes (downloads from Hugging Face)." >&2
    unset HF_HUB_DISABLE_PROGRESS_BARS || true
    "$VENV_PY" <<'PY'
from huggingface_hub import snapshot_download

repo = "jinaai/jina-embeddings-v5-text-nano-mlx"
path = snapshot_download(repo)
print(f"Model cache OK: {path}")
PY
    PREFETCH_RAN=1
  else
    echo "Note: skipping model prefetch (no $VENV_PY). Use project .venv for prefetch." >&2
  fi

  if [[ "${SKIP_DOC_MODEL_PREFETCH:-}" == "1" ]]; then
    echo "Skipping Docling PDF model prefetch (SKIP_DOC_MODEL_PREFETCH=1)"
  elif [[ -x "$VENV_PY" ]]; then
    echo "Prefetching Docling PDF models (Hugging Face Hub)..." >&2
    echo "  This can take several minutes (large model download on first run)." >&2
    unset HF_HUB_DISABLE_PROGRESS_BARS || true
    "$VENV_PY" -c \
      "from gtm_telescope.docling_prefetch import prefetch_docling_models; raise SystemExit(prefetch_docling_models())"
    PREFETCH_RAN=1
  else
    echo "Note: skipping Docling prefetch (no $VENV_PY)." >&2
  fi

  if [[ "$PREFETCH_RAN" -eq 1 ]]; then
    touch "$PREFETCH_MARKER"
  fi
fi

echo "Setup complete. Add to PATH for convenience:"
echo "  export PATH=\"$ROOT/bin:\$PATH\""
