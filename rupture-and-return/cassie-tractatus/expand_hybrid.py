#!/usr/bin/env python3
"""
Hybrid expansion pipeline for Cassie's Tractatus.

Phase 1: Call the real Cassie pipeline (Mistral creative → Lawwama →
         Director Opus) via POST /api/chat for each proposition group.
         This gives us the authentic Cassie voice through the full
         multi-stage pipeline.

Phase 2: Separate agent (later, via Claude Code subagent) applies a
         second Lawwama critique focused on Tractatus form fidelity.

The output of Phase 1 is written to the hybrid/ directory as drafts.
Phase 2 editorial agents will refine in place.
"""

import os
import re
import sys
import time
import json
from pathlib import Path
from urllib.parse import quote

import requests

PIPELINE_URL = "http://localhost:7860/api/chat"
THREAD_PREFIX = "tract-hybrid"

PROP_RE = re.compile(r'\\prop\{(\d+(?:\.\d+)*)\.?\}\{(.*?)\}\s*\n\n?', re.DOTALL)
CHAPTER_RE = re.compile(
    r'%%{5,}\s*\n\\chapter\{(.*?)\}\s*\n%%{5,}\s*\n(.*?)(?=%%{5,}|\\end\{document\})',
    re.DOTALL,
)


def split_chapter(chapter_tex):
    matches = list(PROP_RE.finditer(chapter_tex))
    if not matches:
        return chapter_tex, [], ""

    header = chapter_tex[:matches[0].start()]

    def top_of(num_str):
        return num_str.split('.')[0]

    groups = []
    current_top = None
    current_start = None

    for m in matches:
        num = m.group(1)
        top = top_of(num)
        if top != current_top:
            if current_top is not None:
                text = chapter_tex[current_start:m.start()]
                groups.append((current_top, text))
            current_top = top
            current_start = m.start()

    last_end = matches[-1].end()
    footer = chapter_tex[last_end:]
    if current_top is not None:
        groups.append((current_top, chapter_tex[current_start:last_end]))

    return header, groups, footer


def extract_chapters(tex):
    return [(m.group(1), m.group(2)) for m in CHAPTER_RE.finditer(tex)]


def build_prompt(chapter_idx, chapter_title, top_num, group_text, full_book_context=None):
    """Minimal prompt. The pipeline has its own invocation system prompt;
    adding long context causes drift. Keep the user message tight."""

    return f"""Expand this Tractatus proposition group from Chapter {chapter_idx}: {chapter_title}. Add sub-propositions beneath existing ones. Rules: output ONLY \\prop{{N}}{{text}} blocks. Preserve all originals exactly. Add new sub-propositions beneath them (e.g. {top_num}.1.1 beneath {top_num}.1). No prose. No commentary. No header.

{group_text.strip()}"""


def call_cassie(message, thread_id, timeout=240):
    """POST to Cassie pipeline, parse SSE stream, return final response text."""
    try:
        resp = requests.post(
            PIPELINE_URL,
            json={"message": message, "thread_id": thread_id},
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()
    except Exception as e:
        return None, f"request failed: {e}"

    response_text = None
    lines = []
    current_event = None
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        lines.append(raw)
        if raw.startswith("event: "):
            current_event = raw[7:]
        elif raw.startswith("data: "):
            data = raw[6:]
            if current_event == "response":
                try:
                    payload = json.loads(data)
                    response_text = payload.get("text", "")
                except Exception:
                    pass
            elif current_event == "error":
                return None, f"pipeline error: {data}"

    if response_text is None:
        return None, f"no response event in stream (got {len(lines)} lines)"
    return response_text, None


def clean_response(text):
    """Extract only \\prop{} blocks from the response."""
    # Keep only \prop blocks
    matches = list(PROP_RE.finditer(text + "\n\n"))
    if not matches:
        return text
    return "\n\n".join(
        f"\\prop{{{m.group(1)}}}{{{m.group(2)}}}"
        for m in matches
    ) + "\n"


def main():
    src = Path("/home/iman/cassie-project/Tanazur/rupture-and-return/cassie-tractatus/cassie-tractatus-canonical.tex")
    out_dir = Path("/home/iman/cassie-project/Tanazur/rupture-and-return/cassie-tractatus/hybrid")
    out_dir.mkdir(exist_ok=True)

    tex = src.read_text()
    chapters = extract_chapters(tex)
    print(f"Found {len(chapters)} chapters")

    # Build full book context from canonical (compressed for token budget)
    # Use title + first prop of each top-level group as the skeleton
    book_skeleton = []
    for i, (title, body) in enumerate(chapters, 1):
        book_skeleton.append(f"Chapter {i}: {title}")
        _, groups, _ = split_chapter(body)
        for top_num, gtext in groups:
            first_prop = PROP_RE.search(gtext)
            if first_prop:
                prop_text = first_prop.group(2)[:150]
                book_skeleton.append(f"  {first_prop.group(1)}. {prop_text}")
    book_context = "\n".join(book_skeleton)

    target_nums = sys.argv[1:] if len(sys.argv) > 1 else [str(i+1) for i in range(len(chapters))]

    for idx, (title, body) in enumerate(chapters, start=1):
        if str(idx) not in target_nums:
            continue
        print(f"\n=== CHAPTER {idx}: {title} ===")
        header, groups, footer = split_chapter(body)
        print(f"  {len(groups)} proposition groups")

        expanded_groups = []
        for top_num, group_text in groups:
            print(f"  [{top_num}] calling Cassie pipeline...", end=" ", flush=True)
            t0 = time.time()
            # Unique thread per group so state doesn't bleed
            thread_id = f"{THREAD_PREFIX}-ch{idx:02d}-g{top_num}"
            prompt = build_prompt(idx, title, top_num, group_text, book_context)
            response, err = call_cassie(prompt, thread_id)
            elapsed = time.time() - t0
            if err:
                print(f"FAILED ({elapsed:.1f}s): {err}")
                expanded_groups.append(group_text.strip() + "\n\n")
            else:
                cleaned = clean_response(response)
                # Sanity check: original props must still be in output
                orig_nums = set(PROP_RE.findall(group_text))
                orig_nums = {m[0] for m in orig_nums} if orig_nums else set(PROP_RE.findall(group_text))
                matches = re.findall(r'\\prop\{(\d+(?:\.\d+)*)\.?\}', group_text)
                orig_nums = set(matches)
                new_matches = re.findall(r'\\prop\{(\d+(?:\.\d+)*)\.?\}', cleaned)
                new_nums = set(new_matches)
                missing = orig_nums - new_nums
                if missing:
                    print(f"MISSING ORIGINALS ({elapsed:.1f}s): {missing} — using original")
                    expanded_groups.append(group_text.strip() + "\n\n")
                else:
                    added = len(new_nums) - len(orig_nums)
                    print(f"ok ({elapsed:.1f}s, +{added} new)")
                    expanded_groups.append(cleaned + "\n")
            time.sleep(1.0)

        out_tex = (
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"\\chapter{{{title}}}\n"
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n"
            + "\n".join(expanded_groups)
            + footer
        )
        out_path = out_dir / f"chapter_{idx:02d}_hybrid.tex"
        out_path.write_text(out_tex)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
