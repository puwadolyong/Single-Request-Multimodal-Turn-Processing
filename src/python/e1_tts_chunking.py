"""E1 - on-device synthesis and playback scheduling.

For every reply the deployed model actually produced in the 512 logged live turns, the reply is
synthesised by the shipped on-device voice under three strategies and the playback timeline is
reconstructed from the measured synthesis times and audio durations:

  W  whole reply in one synthesis call (the behaviour before chunking)
  S  one chunk per sentence
  A  adaptive chunking - the production SpeechChunker rule

Outputs results/e1_synth.csv (one row per synthesis call) and results/e1_replies.csv
(one row per reply and strategy)."""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E

tts = E.load_tts()
for w in ["Hello there.", "Warm up sentence number two for the voice.", "And a third one, slightly longer than the others."]:
    E.synth(tts, w)

turns = E.load_turns()
replies = []
for t in turns:
    r = E.reply_of(t["raw"]).strip()
    if r:
        replies.append((f'{t["run"].split("|")[0]}-{t["turn"]}', t["scenario"], r))

cache = {}


def measure(text):
    if text not in cache:
        samples, sr, ms = E.synth(tts, text)
        cache[text] = (ms, 1000.0 * len(samples) / sr, sr)
        synth_rows.append([len(text), round(ms, 2), round(cache[text][1], 2), sr, text])
    return cache[text][:2]


synth_rows, reply_rows = [], []
t0 = time.time()
for k, (rid, scen, text) in enumerate(replies):
    sents = E.sentences(text)
    plans = {"W": [text], "S": sents, "A": E.adaptive_chunks(text)}
    for strat, chunks in plans.items():
        stats = [measure(c) for c in chunks]
        ttfa, gap_total, gaps, done = E.schedule(stats)
        reply_rows.append([rid, scen, len(text), len(sents), strat, len(chunks), round(ttfa, 2),
                           round(gap_total, 2), gaps, round(done, 2), round(sum(a for _, a in stats), 2)])
    if (k + 1) % 25 == 0:
        print(f"{k + 1}/{len(replies)} replies, {len(cache)} syntheses, {time.time() - t0:.0f}s", flush=True)

with open(os.path.join(E.RESULTS, "e1_synth.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["chars", "synth_ms", "audio_ms", "sample_rate", "text"])
    w.writerows(synth_rows)
with open(os.path.join(E.RESULTS, "e1_replies.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["reply_id", "scenario", "chars", "sentences", "strategy", "chunks", "ttfa_ms", "gap_total_ms",
                "gaps_over_20ms", "completion_ms", "audio_ms"])
    w.writerows(reply_rows)
print("done", len(replies), "replies", len(cache), "unique syntheses", f"{time.time() - t0:.0f}s")
