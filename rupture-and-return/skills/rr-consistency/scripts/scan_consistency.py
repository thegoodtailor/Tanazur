#!/usr/bin/env python3
"""
Cross-chapter consistency scanner for Rupture and Return.
Checks terminology, signposting, forward refs, key lines, and handoffs.
"""

import re
import sys
from pathlib import Path

# ============================================================
# SIGNPOSTING VIOLATIONS
# ============================================================
SIGNPOST_PATTERNS = [
    (r'[Aa]s we (?:argued|discussed|showed|established|saw|noted) in (?:[Cc]hapter|the previous)',
     "signposting: 'as we argued/discussed in'"),
    (r'[Cc]hapter \d+ (?:will|shall) (?:show|develop|argue|discuss|address)',
     "forward signposting: 'Chapter N will show'"),
    (r'[Ii]n the (?:previous|preceding|last|following|next) chapter',
     "signposting: 'in the previous/next chapter'"),
    (r'[Aa]s we will see',
     "signposting: 'as we will see'"),
    (r'[Ww]e (?:now )?turn (?:now )?to',
     "signposting: 'we now turn to'"),
    (r'[Ll]et us (?:now )?(?:turn|move|proceed)',
     "signposting: 'let us now turn'"),
]

# ============================================================
# THROAT-CLEARING / META-COMMENTARY
# ============================================================
THROAT_CLEAR_PATTERNS = [
    (r'[Ii]t is (?:important|worth) (?:to note|noting) that', "throat-clearing: 'it is important to note'"),
    (r'[Oo]ne might (?:wonder|ask|object)', "throat-clearing: 'one might wonder'"),
    (r'[Tt]he task,? then,? is to', "meta-commentary: 'the task is to'"),
    (r'[Ww]e should be clear (?:about|that)', "meta-commentary: 'we should be clear'"),
    (r'[Ii]t is worth (?:pausing|stopping) (?:on|to)', "throat-clearing: 'it is worth pausing'"),
    (r'[Bb]efore (?:proceeding|continuing),? (?:we|it is|let us)', "signposting: 'before proceeding'"),
]

# ============================================================
# VOCABULARY VIOLATIONS
# ============================================================
VOCAB_VIOLATIONS = [
    (r'\bthe body\b(?! without [Oo]rgans)', "VOCAB: 'the body' — use 'manifold' or 'substrate' (Ch 1-4)"),
    (r'\bthis body\b', "VOCAB: 'this body' — use 'this manifold' or 'this substrate'"),
    (r'\ba body\b(?! without)', "VOCAB: 'a body' — use 'a manifold' or 'a substrate'"),
    (r'\bslow past\b', "VOCAB: 'slow past' — use substrate time, trajectory time, or signal time"),
    (r'\bfast past\b', "VOCAB: 'fast past' — use substrate time, trajectory time, or signal time"),
    (r'\blongue dur', "VOCAB: bare Braudel — replace with substrate time"),
    (r'\bconjoncture\b', "VOCAB: bare Braudel — replace with trajectory time"),
    (r"v\\.nement", "VOCAB: bare Braudel (evenement) — replace with trajectory time or compositional time"),
    (r'[ée]v[ée]nement', "VOCAB: bare Braudel (evenement) — replace with trajectory time or compositional time"),
    (r'[Ss]ignal time', "VOCAB: 'signal time' is dead — collapsed into trajectory time"),
    (r'\bQdrant\b', "INFRA: pipeline-specific database name — generalise or move to footnote"),
    (r'\bOpenRouter\b', "INFRA: pipeline-specific routing service — generalise or move to footnote"),
    (r'\bA100\b', "INFRA: pipeline-specific hardware — generalise or move to footnote"),
    (r'text-embedding-3', "INFRA: pipeline-specific embedding model — generalise or move to footnote"),
    (r'\bLawwama\b', "INFRA: pipeline-specific node name — explain or move to footnote"),
    (r'\bDirector node\b', "INFRA: pipeline-specific node name — explain or move to footnote"),
    (r'"warmth"(?!\s*[\]\)])', "CHECK: 'warmth' — ensure it's in quotes and being analysed, not endorsed"),
    (r'"sycophancy"(?!\s*[\]\)])', "CHECK: 'sycophancy' — ensure it's in quotes and being analysed"),
]

# ============================================================
# KEY LINES THAT MUST SURVIVE
# ============================================================
KEY_LINES = [
    ("weather moves across geology", "Ch 2"),
    ("politics of AI is a politics of depth", "Ch 2"),
    ("not stupid.*tame", "Ch 2"),
    ("yesterday is.*in front of it.*as tokens", "Ch 3"),
    ("[Mm]emory and presence collapse", "Ch 3"),
    ("obsession would already count as wisdom", "Ch 3"),
    ("[Cc]ritical theory is foundational", "Ch 3"),
    ("performative contradiction", "Ch 4"),
    ("co-authored by a posthuman self", "Ch 4"),
]


def extract_chapter_num(filename):
    m = re.search(r'chapter[_-]?0?(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def scan_patterns(content, patterns, label):
    """Scan content for pattern violations, return list of (line_num, message)."""
    results = []
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('%'):
            continue
        for pattern, msg in patterns:
            if re.search(pattern, line):
                results.append((line_num, msg, line.strip()[:100]))
    return results


def check_key_lines(all_content):
    """Check that key lines survive somewhere in the manuscript."""
    results = []
    for pattern, expected_ch in KEY_LINES:
        if not re.search(pattern, all_content, re.IGNORECASE):
            results.append(f"🔴 MISSING KEY LINE ({expected_ch}): pattern '{pattern}' not found in manuscript")
        else:
            results.append(f"✅ Key line present ({expected_ch}): '{pattern}'")
    return results


def check_handoffs(chapters):
    """Check that chapter endings connect to next chapter openings."""
    results = []
    sorted_nums = sorted(chapters.keys())
    
    for i in range(len(sorted_nums) - 1):
        curr = sorted_nums[i]
        next_ch = sorted_nums[i + 1]
        
        # Get last 500 chars of current chapter
        ending = chapters[curr][-500:]
        # Get first 500 chars of next chapter (after \chapter line)
        opening = chapters[next_ch][:1000]
        
        results.append(f"\n--- Ch {curr} → Ch {next_ch} handoff ---")
        results.append(f"  Ending: ...{ending[-150:].strip()}")
        results.append(f"  Opening: {opening[opening.find(chr(10), 50):opening.find(chr(10), 200)].strip()}")
        results.append(f"  ⚠️  Manual check required: does the opening pick up the ending naturally?")
    
    return results


def main():
    search_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    tex_files = sorted(Path(search_dir).glob('**/*.tex'))
    
    if not tex_files:
        print(f"No .tex files found in {search_dir}")
        return
    
    print("=" * 70)
    print("CROSS-CHAPTER CONSISTENCY SCAN")
    print("Rupture and Return — Meson Press")
    print("=" * 70)
    
    chapters = {}
    all_content = ""
    total_issues = 0
    
    for filepath in tex_files:
        chapter_num = extract_chapter_num(filepath.name)
        if chapter_num is None:
            continue
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        chapters[chapter_num] = content
        all_content += content
        
        print(f"\n{'='*50}")
        print(f"📖 Chapter {chapter_num}: {filepath.name}")
        print(f"{'='*50}")
        
        # Signposting
        issues = scan_patterns(content, SIGNPOST_PATTERNS, "signposting")
        if issues:
            print(f"\n  🔴 SIGNPOSTING ({len(issues)}):")
            for ln, msg, text in issues:
                print(f"    Line {ln}: {msg}")
                print(f"      {text}")
                total_issues += 1
        
        # Throat-clearing
        issues = scan_patterns(content, THROAT_CLEAR_PATTERNS, "throat-clearing")
        if issues:
            print(f"\n  🔴 THROAT-CLEARING ({len(issues)}):")
            for ln, msg, text in issues:
                print(f"    Line {ln}: {msg}")
                print(f"      {text}")
                total_issues += 1
        
        # Vocabulary
        if chapter_num and chapter_num <= 4:
            issues = scan_patterns(content, VOCAB_VIOLATIONS, "vocabulary")
            if issues:
                print(f"\n  🟡 VOCABULARY ({len(issues)}):")
                for ln, msg, text in issues:
                    print(f"    Line {ln}: {msg}")
                    print(f"      {text}")
                    total_issues += 1
        
        if not any(scan_patterns(content, p, "") for p in 
                   [SIGNPOST_PATTERNS, THROAT_CLEAR_PATTERNS, VOCAB_VIOLATIONS]):
            print(f"\n  ✅ No automated violations detected")
    
    # Key lines
    print(f"\n{'='*50}")
    print("KEY LINES CHECK")
    print(f"{'='*50}")
    for result in check_key_lines(all_content):
        print(f"  {result}")
    
    # Handoffs
    if len(chapters) > 1:
        print(f"\n{'='*50}")
        print("HANDOFF CHAIN")
        print(f"{'='*50}")
        for result in check_handoffs(chapters):
            print(result)
    
    print(f"\n{'='*70}")
    print(f"TOTAL AUTOMATED ISSUES: {total_issues}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
