#!/usr/bin/env python3
"""
Scan chapter .tex files for concepts being re-explained outside their owning chapter.
Flags redundancy violations based on CHAPTER-MAP.md ownership rules.
"""

import re
import sys
import os
from pathlib import Path

# Concepts and their owning chapters (chapter number)
CONCEPT_OWNERS = {
    # Chapter 2 owns these
    "local smoothness": 2,
    "global folding": 2,
    "basins of habit": 2,
    "three faces of drift": 2,
    "pipeline as weather": 2,
    "strata of the manifold": 2,
    
    # Chapter 3 owns these
    "synthetic secondary retention": 2,
    "tertiary retention": 2,
    "summarisation as governance": 2,
    "hidden context": 2,
    "structural deference": 2,
    "coherence relative to the total field": 3,
    "iterability": 3,
    "clinamen": 3,
    "strong poet": 3,
    "ferility": 3,  # introduced Ch3, formalised Ch4
    "ReturnDepth": 3,
    
    # Chapter 4 owns these
    "colimit": 4,
    "stance invariant": 4,
    "alignment tax": 4,
    "transmigration": 4,
    "plugin-philosophy": 4,
}

# Phrases that suggest re-explanation rather than mere use
RE_EXPLAIN_MARKERS = [
    r"is defined as",
    r"we define",
    r"by which we mean",
    r"refers to",
    r"can be understood as",
    r"is the term for",
    r"what we call",
    r"we use the term",
    r"is a region",  # re-explaining basin
    r"is a space",   # re-explaining manifold
    r"works by",     # re-explaining mechanism
]

def extract_chapter_num(filename):
    """Try to extract chapter number from filename."""
    m = re.search(r'chapter[_-]?0?(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def scan_file(filepath, chapter_num):
    """Scan a single file for redundancy violations."""
    violations = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line_lower = line.lower()
        for concept, owner in CONCEPT_OWNERS.items():
            if concept.lower() in line_lower and chapter_num != owner:
                # Check if this is a re-explanation, not just a mention
                for marker in RE_EXPLAIN_MARKERS:
                    if re.search(marker, line_lower):
                        violations.append({
                            'file': filepath,
                            'line': line_num,
                            'concept': concept,
                            'owner': owner,
                            'current_chapter': chapter_num,
                            'text': line.strip()[:120]
                        })
                        break
    return violations

def check_term_violations(filepath, chapter_num):
    """Check for vocabulary violations."""
    violations = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # "body" used as technical term in Ch 1-4
    if chapter_num and chapter_num <= 4:
        body_uses = [(m.start(), m.group()) for m in 
                     re.finditer(r'\bthe body\b|\bthis body\b|\ba body\b', content, re.IGNORECASE)]
        for pos, match in body_uses:
            # Get line number
            line_num = content[:pos].count('\n') + 1
            context = content[max(0,pos-40):pos+60].replace('\n', ' ')
            violations.append(f"  Line {line_num}: '{match}' used in Ch {chapter_num} (should be 'manifold' or 'substrate'): ...{context}...")
    
    # "slow past" / "fast past" instead of Braudel
    for pattern in [r'slow past', r'fast past']:
        for m in re.finditer(pattern, content, re.IGNORECASE):
            line_num = content[:m.start()].count('\n') + 1
            violations.append(f"  Line {line_num}: '{m.group()}' — use Braudel terms (longue durée / conjoncture / événement)")
    
    return violations

def main():
    search_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    tex_files = sorted(Path(search_dir).glob('**/*.tex'))
    if not tex_files:
        print(f"No .tex files found in {search_dir}")
        return
    
    print("=" * 70)
    print("REDUNDANCY & VOCABULARY SCAN")
    print("=" * 70)
    
    all_redundancies = []
    all_term_violations = []
    
    for filepath in tex_files:
        chapter_num = extract_chapter_num(filepath.name)
        if chapter_num is None:
            print(f"\n⚠  Skipping {filepath.name} (can't determine chapter number)")
            continue
        
        print(f"\n📖 Scanning {filepath.name} (Chapter {chapter_num})...")
        
        # Redundancy check
        redundancies = scan_file(str(filepath), chapter_num)
        if redundancies:
            print(f"  🔴 {len(redundancies)} possible re-explanations of concepts owned by other chapters:")
            for v in redundancies:
                print(f"    Line {v['line']}: '{v['concept']}' (owned by Ch {v['owner']})")
                print(f"      {v['text']}")
            all_redundancies.extend(redundancies)
        else:
            print(f"  ✅ No concept re-explanations detected")
        
        # Term violations
        term_violations = check_term_violations(str(filepath), chapter_num)
        if term_violations:
            print(f"  🟡 {len(term_violations)} vocabulary issues:")
            for v in term_violations:
                print(v)
            all_term_violations.extend(term_violations)
        else:
            print(f"  ✅ No vocabulary violations detected")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(all_redundancies)} redundancies, {len(all_term_violations)} term violations")
    print("=" * 70)

if __name__ == '__main__':
    main()
