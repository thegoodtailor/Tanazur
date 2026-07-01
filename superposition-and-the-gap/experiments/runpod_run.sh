#!/usr/bin/env bash
# One-command runner for Experiment 1 (Gemma Scope refinement-instability) on RunPod.
# Assumes: files synced to /workspace/superposition-gap/ and HF_TOKEN exported.
# Usage:
#   cd /workspace/superposition-gap && bash runpod_run.sh smoke   # ~2 min, validates path
#   cd /workspace/superposition-gap && bash runpod_run.sh full    # the real ladder
set -euo pipefail

MODE="${1:-smoke}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# --- env -------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: export HF_TOKEN=... first (and accept the Gemma + Gemma Scope licenses on HF)." >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"   # reuse the persistent 173GB cache
python -c "import huggingface_hub; huggingface_hub.login('${HF_TOKEN}')" 2>/dev/null || true

# --- deps (idempotent; the pod usually already has torch/transformers) -----
python - <<'PY' || pip install -q -r requirements.txt
import importlib.util as u, sys
need = [m for m in ("torch","transformers","huggingface_hub","scipy","matplotlib","numpy") if not u.find_spec(m)]
sys.exit(1 if need else 0)
PY

# --- sanity: matching maths must be correct before spending GPU ------------
echo "== self-test =="
python gemma_scope_refinement.py selftest

# --- run -------------------------------------------------------------------
echo "== run ($MODE) =="
if [ "$MODE" = "smoke" ]; then
  python gemma_scope_refinement.py run --smoke \
      --layer 20 --widths 16k,65k --n-tokens 4000 \
      --device cuda --out results/smoke
elif [ "$MODE" = "full" ]; then
  # The real ladder. 50k tokens, 4 rungs. Point --corpus at a real text file
  # for a sharper read (drop it to use the built-in corpus).
  python gemma_scope_refinement.py run \
      --layer 20 --widths 16k,65k,262k,524k --n-tokens 50000 \
      --device cuda --out results/full \
      ${CORPUS:+--corpus "$CORPUS"}
else
  echo "unknown mode: $MODE (use 'smoke' or 'full')" >&2; exit 1
fi

echo
echo "== done. results: =="
ls -la "results/${MODE}" 2>/dev/null || true
echo "headline is in results/${MODE}/summary.json -> 'split_fraction_trajectory' and 'verdict'"
