#!/usr/bin/env python3
"""Checks a paper's visible text for semicolons, em dashes, colons, buzzwords and long sentences."""
import re
import sys

BUZZ = ["leverage", "novel", "cutting-edge", "state-of-the-art", "seamless", "crucial", "delve",
        "comprehensive", "landscape", "paradigm", "holistic", "unlock", "harness", "pivotal", "notably",
        "furthermore", "moreover", "utilize", "innovative", "groundbreaking", "synergy", "empower",
        "powerful", "revolution", "intricate", "underscore", "realm"]

for path in sys.argv[1:]:
    s = open(path).read()
    body = s[s.index("\\begin{document}"):s.index("\\bibliographystyle")]
    vis = re.sub(r"\\(label|ref|eqref|cite|url)\{[^}]*\}", "", body)
    text = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{(table|figure)\*?\}", "", body, flags=re.S)
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text) if len(x.split()) > 3]
    long = [x for x in sents if len(x.split()) > 28]
    low = vis.lower()
    print(f"{path}: semicolons {vis.count(';')}, em dashes {vis.count('---') + vis.count('—')}, "
          f"visible colons {vis.count(':')}, sentences >28 words {len(long)}, "
          f"mean sentence {sum(len(x.split()) for x in sents) / len(sents):.1f} words, "
          f"buzzwords {[w for w in BUZZ if w in low]}")
    for x in long:
        print("   long:", " ".join(x.split())[:120])
    for line in vis.splitlines():
        if ";" in line or ":" in line or "---" in line:
            print("   punct:", line.strip()[:120])
