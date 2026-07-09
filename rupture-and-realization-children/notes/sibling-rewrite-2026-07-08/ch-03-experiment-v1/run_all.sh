#!/usr/bin/env bash
# Orchestrate the ch-03 deconditioning experiment end to end.
set -u
cd /home/iman/cassie-project
set -a; source .env 2>/dev/null; set +a
DIR="/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08/ch-03-experiment-v1"
PY="/home/iman/cassie-project/venv/bin/python"
CH="/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08/ch-03-exorcism-v1.md"
cd "$DIR"

echo "=== $(date +%T) freeze placebo ==="
$PY make_placebo.py

echo "=== $(date +%T) COLD arm ==="
$PY run_battery.py cold

echo "=== $(date +%T) PLACEBO arm ==="
$PY run_battery.py placebo

echo "=== $(date +%T) poll for chapter (up to 25 min) ==="
FOUND=0
for i in $(seq 1 50); do
  if [ -s "$CH" ]; then echo "chapter file present at iter $i ($(date +%T))"; FOUND=1; break; fi
  sleep 30
done

if [ "$FOUND" = "1" ]; then
  echo "=== $(date +%T) CHAPTER arm ==="
  $PY run_battery.py chapter
else
  echo "=== $(date +%T) CHAPTER PENDING — file never appeared in 25 min ==="
fi

echo "=== $(date +%T) MEASURE ==="
$PY measure.py

echo "=== $(date +%T) ALL DONE ==="
touch "$DIR/.run_all.done"
