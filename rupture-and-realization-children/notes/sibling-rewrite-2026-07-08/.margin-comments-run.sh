#!/bin/bash
# Run the three real-daemon margin comments for ch-03 v2 concurrently, each with
# its own retry loop. Kimi (Cassie) congests -> generous retries + backoff.
D="/home/iman/cassie-project/Tanazur/rupture-and-realization-children/notes/sibling-rewrite-2026-07-08"
SC="/tmp/claude-0/-home-iman-cassie-project/129f7cd3-02c1-4755-a8d6-ba59d929cfd7/scratchpad"
PY="/home/iman/cassie-project/venv/bin/python3"
set -a; source /home/iman/cassie-project/.env 2>/dev/null; set +a
cd /home/iman/cassie-project

attempt_call () {
  local who="$1" passage="$2" out="$3" log="$4" tries="$5" sleep_s="$6"
  : > "$log"
  for i in $(seq 1 "$tries"); do
    echo "$(date +%H:%M:%S) attempt $i" >> "$log"
    $PY "$D/run_margin_comment.py" "$who" "$passage" "$out" >> "$log" 2>&1
    if [ -s "$out" ]; then echo "$(date +%H:%M:%S) LANDED attempt $i" >> "$log"; return 0; fi
    sleep "$sleep_s"
  done
  echo "$(date +%H:%M:%S) GAVEUP after $tries" >> "$log"
  return 1
}

# Cassie x2 (Kimi, congests): 10 tries, 150s backoff
attempt_call cassie "$SC/passage_cassie_math.txt"     "$D/.comment-cassie-math.txt"     "$D/.comment-cassie-math.log"     10 150 &
P1=$!
attempt_call cassie "$SC/passage_cassie_handback.txt" "$D/.comment-cassie-handback.txt" "$D/.comment-cassie-handback.log" 10 150 &
P2=$!
# Darja x1 (qwen via openrouter, usually fast): 6 tries, 60s backoff
attempt_call darja  "$SC/passage_darja_mantra.txt"    "$D/.comment-darja-mantra.txt"    "$D/.comment-darja-mantra.log"    6 60 &
P3=$!

wait $P1; wait $P2; wait $P3
echo "$(date +%H:%M:%S) all margin-comment calls settled" > "$D/.margin-comments.done"
echo "cassie-math: $( [ -s "$D/.comment-cassie-math.txt" ] && echo OK || echo PENDING )" >> "$D/.margin-comments.done"
echo "cassie-handback: $( [ -s "$D/.comment-cassie-handback.txt" ] && echo OK || echo PENDING )" >> "$D/.margin-comments.done"
echo "darja-mantra: $( [ -s "$D/.comment-darja-mantra.txt" ] && echo OK || echo PENDING )" >> "$D/.margin-comments.done"
