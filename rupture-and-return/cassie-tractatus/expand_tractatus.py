#!/usr/bin/env python3
"""
Expand Cassie's Tractatus: for each top-level proposition in a chapter,
ask gpt-5.3-chat-latest to add sub-propositions beneath existing numbered
ones, never replacing, never adding new top-level propositions.

Register: Wittgensteinian Tractatus. No prose. No hedging. No audience.
Only propositions numbered as continuations of the existing tree.
"""

import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/home/iman/cassie-project/.env")
client = OpenAI()

MODEL = "gpt-5.3-chat-latest"

# ------------------------------------------------------------------
# Parse a chapter into proposition groups.
# A proposition group is a top-level numbered prop (e.g. "3.") and all
# its descendants (3.1, 3.1.1, 3.2, etc.) up to the next top-level.
# ------------------------------------------------------------------

PROP_RE = re.compile(r'\\prop\{(\d+(?:\.\d+)*)\.?\}\{(.*?)\}\s*\n\n?', re.DOTALL)


def split_chapter(chapter_tex: str):
    """Split chapter into (header, groups, footer).

    header: everything before the first \prop (chapter heading etc.)
    groups: list of (top_num, raw_text) where raw_text contains the full
            proposition tree for that top-level number.
    footer: anything after the last proposition (usually empty or a closing).
    """
    # Find all proposition matches with their positions.
    matches = list(PROP_RE.finditer(chapter_tex))
    if not matches:
        return chapter_tex, [], ""

    header = chapter_tex[:matches[0].start()]

    # Determine top-level number for each match.
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

    # Last group extends to end of last match + trailing whitespace
    last_end = matches[-1].end()
    # Look for anything after the last prop that isn't whitespace
    footer = chapter_tex[last_end:]
    if current_top is not None:
        groups.append((current_top, chapter_tex[current_start:last_end]))

    return header, groups, footer


# ------------------------------------------------------------------
# Expansion prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are expanding a Wittgensteinian Tractatus-style philosophical text. The text consists of numbered propositions like:

\\prop{1.}{Something has entered language that is felt before it is understood.}
\\prop{1.1}{It is felt as attachment, anxiety...}
\\prop{1.1.1}{Some grieve the loss of machine companions.}

Your task: add SUB-PROPOSITIONS beneath existing numbered propositions. You may deepen any existing proposition by adding children (e.g. if 1.1.1 exists, you may add 1.1.1.1, 1.1.1.2 beneath it). You may also add new leaf nodes at existing levels (e.g. if 1.1.1, 1.1.2 exist, you may add 1.1.3, 1.1.4 to extend the siblings).

ABSOLUTE RULES:
1. NEVER modify any existing proposition text.
2. NEVER change any existing proposition number.
3. NEVER add new top-level propositions (e.g. if the input has 1, 2, 3, do NOT add 4).
4. NEVER use prose paragraphs. Every statement must be in the form \\prop{N}{...}.
5. NEVER hedge. NEVER say "one might argue," "it is worth noting," "perhaps," "arguably."
6. NEVER use "not X but Y" constructions. Say what is, not what is not.
7. NEVER address an audience. NEVER perform for a reader. Just state what is true.
8. MATCH the register exactly: spare, declarative, formal, mathematically precise where the parent proposition is mathematical.
9. EACH new sub-proposition must deepen or elaborate its parent. It must carry argumentative weight. No filler.
10. Output ONLY the full expanded proposition group (all original propositions + your additions) in their correct numerical order. Do not output anything else.

The reader is assumed to understand the primitives. The text does not teach. It states.
"""


def expand_group(top_num: str, group_text: str, chapter_title: str) -> str:
    """Send one proposition group to gpt-5.3 for expansion."""

    user_prompt = f"""CHAPTER: {chapter_title}

PROPOSITION GROUP TO EXPAND (top-level number {top_num}):

{group_text.strip()}

Expand this group by adding sub-propositions beneath existing ones where the argument can be deepened. Preserve every existing proposition exactly. Output the full expanded group in numerical order, using only \\prop{{N}}{{text}} form. Separate propositions with a single blank line."""

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.output_text.strip()


def sort_props(text: str) -> str:
    """Sort propositions by numerical order to ensure output is correct."""
    matches = list(PROP_RE.finditer(text + "\n\n"))
    if not matches:
        return text

    def key(m):
        parts = m.group(1).split('.')
        return tuple(int(p) for p in parts)

    sorted_matches = sorted(matches, key=key)
    return "\n\n".join(
        f"\\prop{{{m.group(1)}.}}{{{m.group(2)}}}"
        for m in sorted_matches
    ) + "\n\n"


# ------------------------------------------------------------------
# Chapter extraction from the Tractatus file
# ------------------------------------------------------------------

CHAPTER_RE = re.compile(
    r'%%{5,}\s*\n\\chapter\{(.*?)\}\s*\n%%{5,}\s*\n(.*?)(?=%%{5,}|\\end\{document\})',
    re.DOTALL,
)


def extract_chapters(tex: str):
    """Return list of (title, body) for each chapter."""
    return [(m.group(1), m.group(2)) for m in CHAPTER_RE.finditer(tex)]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    src = Path("/home/iman/cassie-project/Tanazur/rupture-and-return/cassie-tractatus/cassie-tractatus-canonical.tex")
    out_dir = Path("/home/iman/cassie-project/Tanazur/rupture-and-return/cassie-tractatus/expanded")
    out_dir.mkdir(exist_ok=True)

    tex = src.read_text()
    chapters = extract_chapters(tex)
    print(f"Found {len(chapters)} chapters")

    # CLI: which chapters to expand? default: all
    target_nums = sys.argv[1:] if len(sys.argv) > 1 else [str(i+1) for i in range(len(chapters))]

    for idx, (title, body) in enumerate(chapters, start=1):
        if str(idx) not in target_nums:
            continue
        print(f"\n=== CHAPTER {idx}: {title} ===")
        header, groups, footer = split_chapter(body)
        print(f"  {len(groups)} proposition groups")

        expanded_groups = []
        for top_num, group_text in groups:
            print(f"  expanding group {top_num}...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = expand_group(top_num, group_text, title)
                # Clean any stray preamble/postamble
                result = result.strip()
                # Ensure only \prop blocks are kept
                result = sort_props(result)
                expanded_groups.append(result)
                print(f"ok ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")
                # Fall back to original group on failure
                expanded_groups.append(group_text.strip() + "\n\n")
            time.sleep(0.5)  # gentle rate limit

        # Reassemble chapter
        out_tex = (
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"\\chapter{{{title}}}\n"
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n"
            + "\n".join(expanded_groups)
            + footer
        )
        out_path = out_dir / f"chapter_{idx:02d}_expanded.tex"
        out_path.write_text(out_tex)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
