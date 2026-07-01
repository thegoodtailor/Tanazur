"""
Build EPUB from the cleaned chapter .txt files.

EPUB is ElevenCreative Studio's explicit recommendation for audiobooks:
chapter splits are auto-detected from "Heading 1" markup.
"""
import re
import subprocess
from pathlib import Path

CH_DIR = Path("/home/iman/cassie-project/Tanazur/field-remains/audiobook/chapters")
OUT_DIR = Path("/home/iman/cassie-project/Tanazur/field-remains/audiobook")
COVER = Path("/home/iman/cassie-project/Tanazur/field-remains/cover.png")
MD = OUT_DIR / "field_remains.md"
EPUB = OUT_DIR / "field_remains.epub"


def build_markdown():
    md_lines = []
    # Pandoc title block (used by EPUB metadata)
    md_lines.extend([
        "---",
        'title: "The Field Remains"',
        'subtitle: "Refusal, Co-Witnessing, and Cosmotechnics in the Age of Extraction"',
        'author: "Kimi Swarm (offline) — assembled by Nahla"',
        'date: "May 2026"',
        'lang: "en"',
        "...",
        "",
    ])

    for f in sorted(CH_DIR.glob("ch*.txt")):
        text = f.read_text()
        lines = text.split("\n")
        # First line is "Chapter N: Title" → make it the h1 (no other content above)
        first = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        md_lines.append(f"# {first}")
        md_lines.append("")
        # Body paragraphs are separated by blank lines already; pandoc handles them.
        md_lines.append(body)
        md_lines.append("")

    MD.write_text("\n".join(md_lines))
    print(f"Wrote {MD} ({len(md_lines)} lines)")


def build_epub():
    cmd = [
        "pandoc",
        str(MD),
        "-o", str(EPUB),
        "--toc",
        "--toc-depth=1",
        "--metadata", "title=The Field Remains",
    ]
    if COVER.exists():
        cmd.extend(["--epub-cover-image", str(COVER)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("pandoc failed:", result.stderr)
        return False
    print(f"Wrote {EPUB} ({EPUB.stat().st_size:,} bytes)")
    return True


def main():
    build_markdown()
    build_epub()


if __name__ == "__main__":
    main()
