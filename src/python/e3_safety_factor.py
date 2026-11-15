"""E3 - sensitivity of the adaptive chunker to the safety factor alpha in (1).

For each multi-sentence reply of E1 the chunk boundaries are recomputed with alpha in
{1.0, 1.15, 1.3, 1.5}; every chunk text is timed by real synthesis (texts already timed in E1 are
reused from results/e1_synth.csv), and the playback timeline is reconstructed with (4)."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E

ALPHAS = [1.0, 1.15, 1.3, 1.5]
cache = {}
with open(os.path.join(E.RESULTS, "e1_synth.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        cache[r["text"]] = (float(r["synth_ms"]), float(r["audio_ms"]))
tts = None


def measure(text):
    global tts
    if text not in cache:
        if tts is None:
            tts = E.load_tts()
            E.synth(tts, "Warm up sentence for the voice.")
        samples, sr, ms = E.synth(tts, text)
        cache[text] = (ms, 1000.0 * len(samples) / sr)
    return cache[text]


def chunks_for(text, alpha):
    s = E.sentences(text)
    out, cur = [], s[0]
    for nxt in s[1:]:
        if len(cur) * E.SPOKEN_MS_PER_CHAR >= (E.SYNTH_FIXED_MS + len(nxt) * E.SYNTH_MS_PER_CHAR) * alpha:
            out.append(cur)
            cur = nxt
        else:
            cur = cur + " " + nxt
    out.append(cur)
    return out


replies = [E.reply_of(t["raw"]).strip() for t in E.load_turns()]
replies = [r for r in replies if len(E.sentences(r)) > 1]
rows = []
for alpha in ALPHAS:
    for i, text in enumerate(replies):
        ch = chunks_for(text, alpha)
        ttfa, gap, gaps, done = E.schedule([measure(c) for c in ch])
        rows.append([alpha, i, len(ch), round(ttfa, 1), round(gap, 1), gaps])
    print("alpha", alpha, "done", flush=True)
with open(os.path.join(E.RESULTS, "e3_alpha.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["alpha", "reply", "chunks", "ttfa_ms", "gap_total_ms", "gaps_over_20ms"])
    w.writerows(rows)
print("done")
