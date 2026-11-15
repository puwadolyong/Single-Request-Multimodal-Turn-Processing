"""E2b - on-device synthesis time for the replies produced in E2 (run after E2, with the CPU
otherwise idle): time to synthesise the first adaptive chunk and the whole reply."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E

tts = E.load_tts()
for w in ["Hello there.", "Warm up sentence number two for the voice."]:
    E.synth(tts, w)
rows = list(csv.DictReader(open(os.path.join(E.RESULTS, "e2_turns.csv"), encoding="utf-8")))
out = []
for r in rows:
    if r["condition"] not in ("C1", "C2") or r["ok"] != "True" or not r["reply"].strip():
        continue
    chunks = E.adaptive_chunks(r["reply"])
    _, _, first = E.synth(tts, chunks[0])
    _, _, whole = E.synth(tts, r["reply"])
    out.append([r["uid"], r["condition"], len(r["reply"]), len(chunks), round(first, 1), round(whole, 1)])
with open(os.path.join(E.RESULTS, "e2_local_tts.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["uid", "condition", "chars", "chunks", "first_chunk_ms", "whole_ms"])
    w.writerows(out)
print("done", len(out))
