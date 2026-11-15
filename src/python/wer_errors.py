"""Lists the E2 transcripts that differ from the scripted utterance after normalisation."""
import csv
import os
import re
import sys

from num2words import num2words

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def norm(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"\$(\d+)", r"\1 dollars", t)
    t = re.sub(r"\d+(\.\d+)?", lambda m: num2words(float(m.group()) if "." in m.group() else int(m.group())), t)
    return re.sub(r"[^a-z' ]+", " ", t.replace("-", " ")).split()


for r in csv.DictReader(open(os.path.join(HERE, "results", "e2_turns.csv"), encoding="utf-8")):
    if r["condition"] in ("C1", "C2") and norm(r["utterance"]) != norm(r["transcript"]):
        print(r["condition"], "|", r["utterance"], "|", r["transcript"])
