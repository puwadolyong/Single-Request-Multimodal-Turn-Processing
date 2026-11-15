"""Revision Experiment (R2-2 Real Human Non-Native Speech Evaluation on SpeechOcean762):
Selects a deterministic stratified N=100 sample from mispeech/speechocean762 (test split, 2,500 real human L2 speakers)
spanning low (accuracy<=5), mid (6..7), and high (>=8) human expert ratings and adult/youth speakers.
Evaluates:
  1. Whisper small.en (cpu/int8 4t) vs Gemini STT (3.6-flash tb0) vs Gemini Single-Request (FreeTalk scenario, 3.6-flash tb0).
  2. ASR WER (separating punctuation/number formatting vs lexical substitutions).
  3. Pedagogical score alignment against human expert accuracy/fluency/total scores (Spearman rho, Pearson r, mean by stratum).
  4. Pronunciation feedback precision/recall against expert word-level mispronunciation annotations & accuracy<=6,
     plus abstention gating threshold curve (score < tau).
"""
import base64
import io
import json
import math
import os
import random
import re
import statistics
import sys
import time
import faster_whisper
import huggingface_hub
import jiwer
import pyarrow.parquet as pq
import requests
import soundfile as sf
from num2words import num2words

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common

RESULTS_DIR = exp_common.RESULTS
CKPT_PATH = os.path.join(RESULTS_DIR, "rev_e5_speechocean_ckpt.jsonl")
OUT_JSON = os.path.join(RESULTS_DIR, "rev_speechocean762.json")


def norm_raw(s: str) -> str:
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def norm_canon(s: str) -> str:
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"\$(\d+)", lambda m: num2words(int(m.group(1))) + " dollars", s)
    def repl_num(m):
        v = m.group(0)
        return num2words(float(v) if "." in v else int(v))
    s = re.sub(r"\b\d+(?:\.\d+)?\b", repl_num, s)
    s = re.sub(r"[^a-z' ]+", " ", s)
    return " ".join(s.split())


def wer_metrics(refs, hyps, fn):
    rn = [fn(r) or "empty" for r in refs]
    hn = [fn(h) or "empty" for h in hyps]
    out = jiwer.process_words(rn, hn)
    exact = sum(1 for a, b in zip(rn, hn) if a == b)
    rw = out.hits + out.substitutions + out.deletions
    err = out.substitutions + out.deletions + out.insertions
    return {
        "n": len(refs),
        "exact": exact,
        "exact_pct": round(exact / len(refs) * 100.0, 2),
        "ref_words": rw,
        "sub": out.substitutions,
        "del": out.deletions,
        "ins": out.insertions,
        "wer_pct": round(err / max(1, rw) * 100.0, 2),
    }


def spearman_and_pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    pearson = num / max(1e-9, dx * dy)

    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg_r = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                rk[order[k]] = avg_r
            i = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mrx, mry = sum(rx) / n, sum(ry) / n
    r_num = sum((a - mrx) * (b - mry) for a, b in zip(rx, ry))
    r_dx = math.sqrt(sum((a - mrx) ** 2 for a in rx))
    r_dy = math.sqrt(sum((b - mry) ** 2 for b in ry))
    spearman = r_num / max(1e-9, r_dx * r_dy)
    return round(spearman, 4), round(pearson, 4)


def load_ckpt():
    ckpt = {}
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    ckpt[o["ckey"]] = o
    return ckpt


def append_ckpt(rec):
    with open(CKPT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def post_gemini(sess, url, key, body, retries=4):
    hdr = {"Content-Type": "application/json", "x-goog-api-key": key}
    last_err = None
    for a in range(retries):
        try:
            r = sess.post(url, headers=hdr, json=body, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (a + 1))
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(last_err)


def main():
    pq_path = huggingface_hub.hf_hub_download("mispeech/speechocean762", "data/test-00000-of-00001.parquet", repo_type="dataset")
    all_rows = pq.read_table(pq_path).to_pylist()
    rng = random.Random(42)

    # Stratify into Low (acc<=5, n=33), Mid (6<=acc<=7, n=33), High (acc>=8, n=34)
    # Within each stratum prefer balanced youth/adult and mispronunciation representation
    for idx, r in enumerate(all_rows):
        r["_idx"] = idx
        r["_has_misp"] = any(bool(w.get("mispronunciations")) for w in r["words"])

    low_b = [r for r in all_rows if r["accuracy"] <= 5]
    mid_b = [r for r in all_rows if 6 <= r["accuracy"] <= 7]
    hi_b = [r for r in all_rows if r["accuracy"] >= 8]

    def pick_balanced(pool, k):
        adults = [r for r in pool if r["age"] > 15]
        youth = [r for r in pool if r["age"] <= 15]
        rng.shuffle(adults); rng.shuffle(youth)
        half = k // 2
        res = adults[:half] + youth[: k - half]
        if len(res) < k:
            rem = [r for r in pool if r not in res]
            rng.shuffle(rem)
            res += rem[: k - len(res)]
        return res[:k]

    sample = pick_balanced(low_b, 33) + pick_balanced(mid_b, 33) + pick_balanced(hi_b, 34)
    print(f"Selected SpeechOcean762 stratified sample N={len(sample)}: adults={sum(1 for r in sample if r['age']>15)}, with_misp={sum(1 for r in sample if r['_has_misp'])}", flush=True)

    ckpt = load_ckpt()
    key = (os.environ.get("GEMINI_API_KEY") or open(exp_common.KEY_FILE, "r", encoding="utf-8").read().strip())
    ft = exp_common.load_scenarios()["FreeTalk"]
    sys_prompt = exp_common.coach_prompt(ft["persona"], ft["rules"], True)

    # 1. Local Whisper small.en
    need_w = [r for r in sample if f"w_small|{r['_idx']}" not in ckpt]
    if need_w:
        print(f"[SpeechOcean] Running faster-whisper small.en on {len(need_w)} real human L2 wavs...", flush=True)
        wm = faster_whisper.WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
        for i, r in enumerate(need_w):
            arr, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            t0 = time.perf_counter()
            segs, _ = wm.transcribe(arr, beam_size=1, language="en")
            txt = " ".join(s.text.strip() for s in segs).strip()
            ms = (time.perf_counter() - t0) * 1000.0
            rec = {"ckey": f"w_small|{r['_idx']}", "idx": r["_idx"], "ms": ms, "transcript": txt}
            ckpt[rec["ckey"]] = rec
            append_ckpt(rec)

    # 2 & 3. Gemini STT & Gemini Single-Request C1 (3.6-flash tb0)
    sess = requests.Session()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{exp_common.MODEL}:generateContent"
    print(f"[SpeechOcean] Running Gemini STT + Single-Request C1 across N={len(sample)} recordings...", flush=True)
    for i, r in enumerate(sample):
        rid = r["_idx"]
        wb = base64.b64encode(r["audio"]["bytes"]).decode("ascii")
        k_stt = f"gem_stt|{rid}"
        if k_stt not in ckpt:
            time.sleep(0.15)
            body = {
                "contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": wb}}, {"text": exp_common.TRANSCRIBE_PROMPT}]}],
                "generationConfig": {"temperature": exp_common.TRANSCRIBE_TEMPERATURE, "maxOutputTokens": exp_common.TRANSCRIBE_MAX_TOKENS, "thinkingConfig": {"thinkingBudget": 0}},
            }
            t0 = time.perf_counter()
            j = post_gemini(sess, url, key, body)
            ms = (time.perf_counter() - t0) * 1000.0
            txt = "".join(p.get("text", "") for p in j.get("candidates", [{}])[0].get("content", {}).get("parts", []))
            rec = {"ckey": k_stt, "idx": rid, "ms": ms, "transcript": txt.strip()}
            ckpt[k_stt] = rec
            append_ckpt(rec)

        k_c1 = f"gem_c1|{rid}"
        if k_c1 not in ckpt:
            time.sleep(0.15)
            body = {
                "systemInstruction": {"parts": [{"text": sys_prompt}]},
                "contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": wb}}, {"text": "Evaluate the English learner's spoken response."}]}],
                "generationConfig": {
                    "temperature": exp_common.TEMPERATURE,
                    "maxOutputTokens": 2048,
                    "thinkingConfig": {"thinkingBudget": 0},
                    "responseMimeType": "application/json",
                    "responseSchema": exp_common.SCHEMA,
                },
            }
            t0 = time.perf_counter()
            j = post_gemini(sess, url, key, body)
            ms = (time.perf_counter() - t0) * 1000.0
            raw = "".join(p.get("text", "") for p in j.get("candidates", [{}])[0].get("content", {}).get("parts", []))
            try:
                parsed = json.loads(raw)
                ok = True
            except Exception:
                parsed = {}
                ok = False
            rec = {
                "ckey": k_c1, "idx": rid, "ms": ms, "ok": ok,
                "transcript": parsed.get("transcription", ""),
                "score": parsed.get("score"), "status": parsed.get("status"),
                "pronunciation": parsed.get("pronunciation", ""),
                "correction": parsed.get("correction", ""),
                "reply": parsed.get("reply", ""),
            }
            ckpt[k_c1] = rec
            append_ckpt(rec)
        if (i + 1) % 20 == 0:
            print(f"  [SpeechOcean] {i+1}/{len(sample)} done", flush=True)

    # Aggregate results
    refs = [r["text"] for r in sample]
    w_hyps = [ckpt[f"w_small|{r['_idx']}"]["transcript"] for r in sample]
    g_stt_hyps = [ckpt[f"gem_stt|{r['_idx']}"]["transcript"] for r in sample]
    g_c1_recs = [ckpt[f"gem_c1|{r['_idx']}"] for r in sample]
    g_c1_hyps = [x["transcript"] for x in g_c1_recs]

    # Calibration & score stratification
    valid_pairs = [(r, x) for r, x in zip(sample, g_c1_recs) if isinstance(x.get("score"), (int, float))]
    preds = [x["score"] for _, x in valid_pairs]
    accs = [r["accuracy"] for r, _ in valid_pairs]
    flus = [r["fluency"] for r, _ in valid_pairs]
    tots = [r["total"] for r, _ in valid_pairs]
    rho_acc, r_acc = spearman_and_pearson(preds, accs)
    rho_flu, r_flu = spearman_and_pearson(preds, flus)
    rho_tot, r_tot = spearman_and_pearson(preds, tots)

    strata = {}
    for sname, lo, hi in [("low_acc_le5", 0, 5), ("mid_acc_6_7", 6, 7), ("high_acc_ge8", 8, 10)]:
        sub = [x["score"] for r, x in valid_pairs if lo <= r["accuracy"] <= hi]
        strata[sname] = {
            "n": len(sub),
            "convia_score_mean": round(statistics.mean(sub), 2) if sub else None,
            "convia_score_med": round(statistics.median(sub), 2) if sub else None,
        }

    # Pronunciation flag precision/recall + abstention threshold curve
    # Expert ground truth labels: (A) accuracy <= 6, (B) word-level mispronunciations non-empty
    def pron_eval(gt_fn, tau_cut=None):
        tp = fp = fn = tn = 0
        for r, x in valid_pairs:
            gt = gt_fn(r)
            has_pron = bool((x.get("pronunciation") or "").strip())
            if tau_cut is not None and x["score"] >= tau_cut:
                pred = False  # abstain emitting pronunciation flag when overall score >= tau
            else:
                pred = has_pron
            if pred and gt: tp += 1
            elif pred and not gt: fp += 1
            elif not pred and gt: fn += 1
            else: tn += 1
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}

    abstention_acc6 = {str(tau or "ungated"): pron_eval(lambda r: r["accuracy"] <= 6, tau) for tau in [None, 95, 90, 85, 80, 75]}
    abstention_misp = {str(tau or "ungated"): pron_eval(lambda r: r["_has_misp"], tau) for tau in [None, 95, 90, 85, 80, 75]}

    out = {
        "sample_n": len(sample),
        "adults_n": sum(1 for r in sample if r["age"] > 15),
        "youth_n": sum(1 for r in sample if r["age"] <= 15),
        "with_word_mispronunciation_n": sum(1 for r in sample if r["_has_misp"]),
        "accuracy_le_6_n": sum(1 for r in sample if r["accuracy"] <= 6),
        "asr_comparison": {
            "whisper_small_en": {
                "latency_med_ms": round(statistics.median(ckpt[f"w_small|{r['_idx']}"]["ms"] for r in sample), 1),
                "raw": wer_metrics(refs, w_hyps, norm_raw),
                "canon": wer_metrics(refs, w_hyps, norm_canon),
            },
            "gemini_3_6_stt_tb0": {
                "latency_med_ms": round(statistics.median(ckpt[f"gem_stt|{r['_idx']}"]["ms"] for r in sample), 1),
                "raw": wer_metrics(refs, g_stt_hyps, norm_raw),
                "canon": wer_metrics(refs, g_stt_hyps, norm_canon),
            },
            "gemini_3_6_single_request_tb0": {
                "json_valid": sum(1 for x in g_c1_recs if x["ok"]),
                "latency_med_ms": round(statistics.median(x["ms"] for x in g_c1_recs), 1),
                "raw": wer_metrics(refs, g_c1_hyps, norm_raw),
                "canon": wer_metrics(refs, g_c1_hyps, norm_canon),
            },
        },
        "human_score_alignment": {
            "n": len(valid_pairs),
            "spearman_vs_accuracy": rho_acc,
            "pearson_vs_accuracy": r_acc,
            "spearman_vs_fluency": rho_flu,
            "pearson_vs_fluency": r_flu,
            "spearman_vs_total": rho_tot,
            "pearson_vs_total": r_tot,
            "strata": strata,
        },
        "pronunciation_abstention_vs_acc_le6": abstention_acc6,
        "pronunciation_abstention_vs_word_misp": abstention_misp,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
