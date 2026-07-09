#!/bin/bash
# Downstream v2 pipeline: assemble completed chapter -> run v2 CHAPTER arm ->
# judge + embed + compare against v1/cold/placebo -> report.
set -u
D="/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08"
V2="$D/ch-03-experiment-v2"
PY="/home/iman/cassie-project/venv/bin/python3"
cd /home/iman/cassie-project
set -a; source .env 2>/dev/null; set +a

echo "=== $(date +%T) ASSEMBLE chapter (fill comment boxes) ==="
$PY "$D/assemble_chapter.py" || { echo "assemble failed"; exit 1; }

echo "=== $(date +%T) RUN v2 CHAPTER arm ==="
cd "$V2"
$PY run_battery.py chapter

echo "=== $(date +%T) COMPARE (judge v2 + embed v2 + report) ==="
$PY compare.py

echo "=== $(date +%T) V2 PIPELINE DONE ==="
touch "$V2/.run_v2.done"
