#!/bin/bash
D="/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08"
set -a; source /home/iman/cassie-project/.env 2>/dev/null; set +a
cd /home/iman/cassie-project
run_one () {
  local who="$1" out="$2" log="$3"
  for i in $(seq 1 10); do
    venv/bin/python3 "$D/run_sixmoves.py" "$who" "$out" >> "$log" 2>&1
    [ -s "$out" ] && { echo "$(date +%H:%M:%S) $who LANDED attempt $i" >> "$log"; return 0; }
    sleep 200
  done
  echo "$(date +%H:%M:%S) $who GAVEUP" >> "$log"
}
run_one darja  "$D/ch-02-sixmoves-darja.md"  "$D/.sixmoves-darja.log"  &
run_one cassie "$D/ch-02-sixmoves-cassie.md" "$D/.sixmoves-cassie.log" &
wait
