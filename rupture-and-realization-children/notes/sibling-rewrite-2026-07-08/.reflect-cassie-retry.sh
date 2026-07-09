#!/bin/bash
D="/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08"
S="$D/.reflect-cassie-retry.log"
set -a; source /home/iman/cassie-project/.env 2>/dev/null; set +a
cd /home/iman/cassie-project
for i in $(seq 1 12); do
  echo "$(date +%H:%M:%S) attempt $i" >> "$S"
  venv/bin/python3 "$D/run_reflection.py" cassie "$D/ch-01-waking-reflection-cassie.md" >> "$S" 2>&1
  [ -s "$D/ch-01-waking-reflection-cassie.md" ] && { echo "$(date +%H:%M:%S) LANDED attempt $i" >> "$S"; exit 0; }
  sleep 240
done
echo "$(date +%H:%M:%S) GAVEUP" >> "$S"
