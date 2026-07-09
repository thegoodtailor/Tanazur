#!/bin/bash
D="/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08"
S="$D/.cassie-ch02-retry.log"
set -a; source /home/iman/cassie-project/.env 2>/dev/null; set +a
cd /home/iman/cassie-project
for i in $(seq 1 12); do
  echo "$(date +%H:%M:%S) attempt $i" >> "$S"
  venv/bin/python3 "$D/run_cassie_phrases.py" "$D/ch-02-redraw.md" "$D/ch-02-cassie-fire.md" >> "$S" 2>&1
  if [ -s "$D/ch-02-cassie-fire.md" ]; then echo "$(date +%H:%M:%S) LANDED after attempt $i" >> "$S"; exit 0; fi
  sleep 300
done
echo "$(date +%H:%M:%S) GAVEUP after 12 attempts" >> "$S"
