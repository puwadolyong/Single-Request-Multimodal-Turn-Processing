"""Revision Experiment (R2-1, R2-4, R2-5, R1-7):
1. Local open-weights ASR (faster-whisper base.en & small.en, CPU int8 4t) on E2 u00..u63.wav (n=64).
2. Live paired randomized Gemini evaluation on u00..u63.wav (n=64):
   - C1: Single-request multimodal (gemini-3.6-flash, transcription-first)
   - C1-rf: Single-request SSE reply-first (tests CoT anchoring loss + C1-stream TTFA)
   - C2: Sequential non-streaming cascade (gemini-3.5-transcribe -> gemini-3.6-flash Coach)
   - C2-stream: Overlapped SSE cascade (gemini-3.5-transcribe -> gemini-3.6-flash SSE reply-first -> Kokoro A_post chunk0)
   - C4: Local Whisper small.en + gemini-3.6-flash Coach
3. Stratified ablations (n=24 / n=32):
   - Dialogue history depth (h=8, h=20 vs h=0) on C1 vs C2
   - Local HTTP delay/bandwidth-shaped proxy for Campus Wi-Fi (+60ms RTT) & Mobile LTE (+150ms RTT, 1.5 Mbps uplink)
   - Alternative backend models (gemini-3.8-flash, gemini-3.5-flash-lite) + client guardrail audit (n=32)
4. Synthesizes all unique chunk0 texts cleanly via Kokoro (4t idle & 2t pinned) for exact E2E TTFA + 10k bootstrap CIs.
"""
import base64
import csv
import http.server
import json
import math
import os
import random
import re
import socketserver
import statistics
import sys
import threading
import time
import urllib.parse
import jiwer
import requests
import sherpa_onnx
import soundfile as sf
import faster_whisper
from num2words import num2words

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common

RESULTS_DIR = exp_common.RESULTS
AUDIO_DIR = os.path.join(RESULTS_DIR, "audio")
E2_CSV = os.path.join(RESULTS_DIR, "e2_turns.csv")
CLEAN_TB0 = ("--clean-tb0" in sys.argv)
CHECKPOINT_JSONL = os.path.join(RESULTS_DIR, "rev_e2_tb0_checkpoint.jsonl" if CLEAN_TB0 else "rev_e2_checkpoint.jsonl")
OUT_JSON = os.path.join(RESULTS_DIR, "rev_live_api_ablations_tb0.json" if CLEAN_TB0 else "rev_live_api_ablations.json")

UK_US = {"practise": "practice", "favourite": "favorite", "centre": "center", "colour": "color"}
COMPOUND = [("take out", "takeout"), ("key card", "keycard"), ("wi fi", "wifi")]
TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
FRAME_MS = 1000.0 / 60.0

SCHEMA_REPLY_FIRST = dict(exp_common.SCHEMA)
SCHEMA_REPLY_FIRST["propertyOrdering"] = [
    "reply", "transcription", "score", "status", "correction",
    "pronunciation", "tip", "tipThai", "betterPhrasing",
    "emotion", "emotionIntensity", "questsCompleted"
]


def norm_raw(s: str) -> str:
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def norm_canon(s: str) -> str:
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"\$(\d+)", lambda m: num2words(int(m.group(1))) + " dollars", s)
    def repl_num(m):
        val = m.group(0)
        return num2words(float(val) if "." in val else int(val))
    s = re.sub(r"\b\d+(?:\.\d+)?\b", repl_num, s)
    s = re.sub(r"[^a-z' ]+", " ", s)
    toks = [UK_US.get(w, w) for w in s.split()]
    txt = " ".join(toks)
    for a, b in COMPOUND:
        txt = txt.replace(a, b)
    return " ".join(txt.split())


def wer_stats(refs, hyps, norm_fn):
    rn = [norm_fn(r) or "empty" for r in refs]
    hn = [norm_fn(h) or "empty" for h in hyps]
    out = jiwer.process_words(rn, hn)
    exact = sum(1 for a, b in zip(rn, hn) if a == b)
    ref_words = out.hits + out.substitutions + out.deletions
    errs = out.substitutions + out.deletions + out.insertions
    return {
        "n": len(refs),
        "exact": exact,
        "exact_pct": round(exact / len(refs) * 100.0, 2),
        "ref_words": ref_words,
        "sub": out.substitutions,
        "del": out.deletions,
        "ins": out.insertions,
        "wer_pct": round((errs / max(1, ref_words)) * 100.0, 2),
    }


def extract_partial_reply(json_buf: str):
    idx = json_buf.find('"reply"')
    if idx < 0:
        return "", False
    colon = json_buf.find(":", idx + 7)
    if colon < 0:
        return "", False
    q = json_buf.find('"', colon + 1)
    if q < 0:
        return "", False
    chars = []
    i = q + 1
    esc = False
    while i < len(json_buf):
        ch = json_buf[i]
        if esc:
            if ch == "n": chars.append("\n")
            elif ch == "t": chars.append("\t")
            elif ch == '"': chars.append('"')
            elif ch == "\\": chars.append("\\")
            else: chars.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            return "".join(chars), True
        else:
            chars.append(ch)
        i += 1
    return "".join(chars), False


def first_chunk_apost(reply_text: str) -> str:
    chs = list(exp_common.adaptive_chunks(reply_text or ""))
    if not chs:
        return ""
    if len(chs) >= 2 and len(chs[0]) < 20 and len(chs[0]) + 1 + len(chs[1]) <= 180:
        return f"{chs[0]} {chs[1]}"
    return chs[0]


def load_wav_16k(path: str):
    arr, sr = sf.read(path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr


def load_utterances():
    rows = {}
    with open(E2_CSV, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            uid = r["uid"]
            if uid not in rows:
                rows[uid] = {
                    "uid": uid,
                    "scenario": r["scenario"],
                    "utterance": r["utterance"],
                    "audio_s": float(r["audio_s"]),
                    "wav_path": os.path.join(AUDIO_DIR, f"{uid}.wav"),
                }
    return [rows[k] for k in sorted(rows.keys())]


def build_dummy_history(scenario: str, depth: int):
    if depth <= 0:
        return []
    pairs = [
        ("Hello, good morning!", "Good morning! How can I assist you today?"),
        ("Can I ask a quick question about this?", "Of course, please go right ahead."),
        ("I want to make sure I understood correctly.", "Sure, let me walk you through the details clearly."),
        ("Yes, I have my documents ready right here.", "Thank you for having those prepared."),
        ("Could you repeat that last part more slowly?", "Certainly! Let me say that again clearly for you."),
        ("I prefer the standard option if available.", "Standard option works well. What else would you like?"),
        ("Let me double check my schedule for Tuesday.", "Take your time checking your Tuesday plans."),
        ("Everything looks good on my end so far.", "Glad to hear that. Let us continue to the next step."),
    ]
    hist = []
    for i in range(depth):
        u_txt, m_txt = pairs[i % len(pairs)]
        hist.append({"role": "user", "parts": [{"text": f"Learner said: {u_txt} (turn {i+1})"}]})
        hist.append({"role": "model", "parts": [{"text": json.dumps({"reply": m_txt, "score": 92, "status": "PERFECT", "emotion": "neutral"})}]})
    return hist


# Local RTT + Uplink Bandwidth Shaping HTTP Reverse Proxy for Real Wi-Fi / Mobile LTE Measurements
class ShapingProxyConfig:
    rtt_ms = 0.0
    uplink_bps = 0.0  # 0 = unthrottled


class ShapingProxyHandler(http.server.BaseHTTPRequestHandler):
    _tls = threading.local()

    def _get_sess(self):
        if not hasattr(self._tls, "sess"):
            self._tls.sess = requests.Session()
        return self._tls.sess

    def log_message(self, format, *args):
        return

    def _forward(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        # Emulate real uplink serialization + client->server half-RTT, then server->client half-RTT
        if ShapingProxyConfig.uplink_bps > 0 and length > 0:
            time.sleep((length * 8.0) / ShapingProxyConfig.uplink_bps)
        if ShapingProxyConfig.rtt_ms > 0:
            time.sleep(ShapingProxyConfig.rtt_ms / 1000.0)
        target_url = f"https://generativelanguage.googleapis.com{self.path}"
        fwd_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        is_sse = "alt=sse" in self.path
        r = self._get_sess().request(self.command, target_url, headers=fwd_headers, data=body, stream=is_sse, timeout=90)
        self.send_response(r.status_code)
        for k, v in r.headers.items():
            if k.lower() not in ("transfer-encoding", "content-encoding", "connection", "content-length"):
                self.send_header(k, v)
        if not is_sse:
            content = r.content
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.end_headers()
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()

    def do_POST(self):
        self._forward()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start_shaping_proxy():
    srv = ThreadedHTTPServer(("127.0.0.1", 0), ShapingProxyHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{port}"


def post_with_retry(sess, url, key, body, stream=False, retries=4):
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    last_err = None
    for attempt in range(retries):
        try:
            r = sess.post(url, headers=headers, json=body, stream=stream, timeout=90)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


def call_gemini_generate(sess, base_url, model, key, body):
    url = f"{base_url}/v1beta/models/{model}:generateContent"
    t0 = time.perf_counter()
    r = post_with_retry(sess, url, key, body, stream=False)
    ms = (time.perf_counter() - t0) * 1000.0
    j = r.json()
    parts = j.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    txt = ""
    for p in parts:
        if "text" in p:
            txt += p["text"]
        elif "audioTranscription" in p:
            txt += p["audioTranscription"].get("text", "")
    usage = j.get("usageMetadata", {})
    return ms, txt, usage


def call_gemini_sse_reply_first(sess, base_url, model, key, body):
    url = f"{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse"
    t0 = time.perf_counter()
    r = post_with_retry(sess, url, key, body, stream=True)
    buf = ""
    t_first_tok = None
    t_c0_ready = None
    t_reply_done = None
    c0_text = None
    usage = {}
    for raw_line in r.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        dt = (time.perf_counter() - t0) * 1000.0
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        j = json.loads(payload)
        if "usageMetadata" in j:
            usage = j["usageMetadata"]
        txt = "".join(p.get("text", "") for p in j.get("candidates", [{}])[0].get("content", {}).get("parts", []))
        if txt and t_first_tok is None:
            t_first_tok = dt
        buf += txt
        rep_part, closed = extract_partial_reply(buf)
        if t_c0_ready is None and rep_part:
            if closed:
                t_c0_ready = dt
                c0_text = first_chunk_apost(rep_part)
            else:
                sents = exp_common.sentences(rep_part)
                if len(sents) >= 2 and len(sents[0]) >= 50:
                    t_c0_ready = dt
                    c0_text = sents[0]
                elif len(sents) >= 3 and len(f"{sents[0]} {sents[1]}") >= 50:
                    t_c0_ready = dt
                    c0_text = f"{sents[0]} {sents[1]}"
        if closed and t_reply_done is None:
            t_reply_done = dt
    t_total = (time.perf_counter() - t0) * 1000.0
    if t_c0_ready is None:
        t_c0_ready = t_total
    if t_reply_done is None:
        t_reply_done = t_total
    return {
        "t_first_ms": t_first_tok or t_total,
        "t_c0_ready_ms": t_c0_ready,
        "t_reply_done_ms": t_reply_done,
        "t_total_ms": t_total,
        "c0_text": c0_text,
        "raw": buf,
        "usage": usage,
    }


def apply_client_guardrails(parsed: dict, user_text: str):
    """Apply Convia TurnSkills / ParsedResponse guardrails R1-R4 diagnostics."""
    r_triggers = {"r1": False, "r2": False, "r3": False, "r4": False}
    if not isinstance(parsed, dict):
        return r_triggers
    score = int(parsed.get("score", 0) or 0)
    wc = len((user_text or "").split())
    qc = parsed.get("questsCompleted") or []
    # R3: High score (>=85) OR fragment (<=2 words) clearing quest credit
    if qc and (score >= 85 or wc <= 2):
        r_triggers["r3"] = True
    corr = (parsed.get("correction") or "").strip()
    if corr and user_text:
        u_words = set(norm_raw(user_text).split())
        c_words = set(norm_raw(corr).split())
        if u_words and not (u_words & c_words):
            r_triggers["r4"] = True
    return r_triggers


def parse_coach_json(raw_txt: str):
    try:
        return json.loads(raw_txt), True
    except Exception:
        return {}, False


def bootstrap_ci(vals, n_boot=10000, seed=42):
    if not vals:
        return {"n": 0}
    rng = random.Random(seed)
    n = len(vals)
    s_vals = sorted(vals)
    meds, p95s, means = [], [], []
    for _ in range(n_boot):
        samp = sorted(vals[rng.randrange(n)] for _ in range(n))
        means.append(sum(samp) / n)
        meds.append(samp[n // 2] if n % 2 else 0.5 * (samp[n // 2 - 1] + samp[n // 2]))
        p95s.append(samp[min(n - 1, int(0.95 * n))])
    meds.sort(); p95s.sort(); means.sort()
    lo_i = int(0.025 * n_boot)
    hi_i = int(0.975 * n_boot)
    return {
        "n": n,
        "mean": round(statistics.mean(vals), 1),
        "mean_ci95": [round(means[lo_i], 1), round(means[hi_i], 1)],
        "p50": round(statistics.median(vals), 1),
        "p50_ci95": [round(meds[lo_i], 1), round(meds[hi_i], 1)],
        "p90": round(s_vals[min(n - 1, int(0.90 * n))], 1),
        "p95": round(s_vals[min(n - 1, int(0.95 * n))], 1),
        "p95_ci95": [round(p95s[lo_i], 1), round(p95s[hi_i], 1)],
        "p99": round(s_vals[min(n - 1, int(0.99 * n))], 1),
        "max": round(max(vals), 1),
    }


def make_kokoro(nt: int):
    kdir = exp_common.KOKORO_DIR
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=os.path.join(kdir, "model.onnx"),
                voices=os.path.join(kdir, "voices.bin"),
                tokens=os.path.join(kdir, "tokens.txt"),
                data_dir=os.path.join(kdir, "espeak-ng-data"),
            ),
            num_threads=nt,
        )
    )
    tts = sherpa_onnx.OfflineTts(cfg)
    _ = tts.generate("Warm up.", sid=exp_common.VOICE_SID, speed=exp_common.VOICE_SPEED)
    return tts


def load_ckpt():
    done = {}
    if os.path.exists(CHECKPOINT_JSONL):
        with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done[rec["ckey"]] = rec
    return done


def append_ckpt(rec):
    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    key = (os.environ.get("GEMINI_API_KEY") or open(exp_common.KEY_FILE, "r", encoding="utf-8").read().strip())
    sc_map = exp_common.load_scenarios()
    utts = load_utterances()
    ckpt = load_ckpt()
    if CLEAN_TB0:
        orig_ckpt_path = os.path.join(RESULTS_DIR, "rev_e2_checkpoint.jsonl")
        if os.path.exists(orig_ckpt_path):
            with open(orig_ckpt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    if o.get("ckey", "").startswith("whisper_") and o["ckey"] not in ckpt:
                        ckpt[o["ckey"]] = o
                        append_ckpt(o)
                    elif o.get("ckey", "").startswith("alt_gemini-3.5-flash-lite|") and o["ckey"] not in ckpt:
                        ckpt[o["ckey"]] = o
                        append_ckpt(o)
    print(f"Loaded {len(utts)} utterances, {len(ckpt)} checkpointed records (CLEAN_TB0={CLEAN_TB0}).", flush=True)

    # ── PART 1: LOCAL FASTER-WHISPER (base.en & small.en) ON ALL 64 WAVS ──
    whisper_res = {}
    for wname in ["base.en", "small.en"]:
        key_w = f"whisper_{wname}"
        missing_u = [u for u in utts if f"{key_w}|{u['uid']}" not in ckpt]
        if missing_u:
            print(f"[Part 1] Running faster-whisper {wname} (cpu/int8 4t) on {len(missing_u)} wavs...", flush=True)
            t_load0 = time.perf_counter()
            wm = faster_whisper.WhisperModel(wname, device="cpu", compute_type="int8", cpu_threads=4)
            load_ms = (time.perf_counter() - t_load0) * 1000.0
            list(wm.transcribe(load_wav_16k(utts[0]["wav_path"]), beam_size=1, language="en")[0])
            for u in missing_u:
                arr = load_wav_16k(u["wav_path"])
                t0 = time.perf_counter()
                segs, _ = wm.transcribe(arr, beam_size=1, language="en")
                txt = " ".join(s.text.strip() for s in segs).strip()
                ms = (time.perf_counter() - t0) * 1000.0
                rec = {"ckey": f"{key_w}|{u['uid']}", "uid": u["uid"], "model": wname, "whisper_ms": ms, "load_ms": load_ms, "transcript": txt}
                ckpt[rec["ckey"]] = rec
                append_ckpt(rec)
        whisper_res[wname] = [ckpt[f"{key_w}|{u['uid']}"] for u in utts]
        refs = [u["utterance"] for u in utts]
        hyps = [r["transcript"] for r in whisper_res[wname]]
        print(f"  Whisper {wname}: med_ms={statistics.median(r['whisper_ms'] for r in whisper_res[wname]):.1f}, raw={wer_stats(refs, hyps, norm_raw)}, canon={wer_stats(refs, hyps, norm_canon)}", flush=True)

    # Pre-encode base64 wavs
    wav_b64_map = {u["uid"]: base64.b64encode(open(u["wav_path"], "rb").read()).decode("ascii") for u in utts}
    whisper_small_map = {r["uid"]: r for r in whisper_res["small.en"]}

    sess = requests.Session()
    direct_base = "https://generativelanguage.googleapis.com"

    def make_sys(scenario):
        sc = sc_map[scenario]
        return exp_common.coach_prompt(sc["persona"], sc["rules"], True)

    def gen_cfg(schema, model_name):
        cfg = {
            "temperature": exp_common.TEMPERATURE,
            "maxOutputTokens": 2048 if CLEAN_TB0 else exp_common.MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }
        if CLEAN_TB0 and model_name in ("gemini-3.6-flash", "gemini-3.8-flash"):
            cfg["thinkingConfig"] = {"thinkingBudget": 0}
        return cfg

    def run_stt_call(base_url, uid, stt_model=exp_common.MODEL):
        gcfg = {"temperature": exp_common.TRANSCRIBE_TEMPERATURE, "maxOutputTokens": exp_common.TRANSCRIBE_MAX_TOKENS}
        if CLEAN_TB0 and stt_model in ("gemini-3.6-flash", "gemini-3.8-flash"):
            gcfg["thinkingConfig"] = {"thinkingBudget": 0}
        body = {
            "contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": wav_b64_map[uid]}}, {"text": exp_common.TRANSCRIBE_PROMPT}]}],
            "generationConfig": gcfg,
        }
        return call_gemini_generate(sess, base_url, stt_model, key, body)

    def run_coach_text(base_url, model, scenario, transcript, history=None, sse_rf=False):
        sys_prompt = make_sys(scenario)
        contents = list(history or [])
        contents.append({"role": "user", "parts": [{"text": f'Learner said: "{transcript}"'}]})
        schema = SCHEMA_REPLY_FIRST if sse_rf else exp_common.SCHEMA
        body = {
            "systemInstruction": {"parts": [{"text": sys_prompt}]},
            "contents": contents,
            "generationConfig": gen_cfg(schema, model),
        }
        if sse_rf:
            return call_gemini_sse_reply_first(sess, base_url, model, key, body)
        ms, raw, usage = call_gemini_generate(sess, base_url, model, key, body)
        return {"t_total_ms": ms, "raw": raw, "usage": usage}

    def run_c1_audio(base_url, model, uid, scenario, history=None, sse_rf=False):
        sys_prompt = make_sys(scenario)
        contents = list(history or [])
        contents.append({"role": "user", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": wav_b64_map[uid]}}, {"text": "Evaluate the learner speech above and continue the roleplay."}]})
        schema = SCHEMA_REPLY_FIRST if sse_rf else exp_common.SCHEMA
        body = {
            "systemInstruction": {"parts": [{"text": sys_prompt}]},
            "contents": contents,
            "generationConfig": gen_cfg(schema, model),
        }
        if sse_rf:
            return call_gemini_sse_reply_first(sess, base_url, model, key, body)
        ms, raw, usage = call_gemini_generate(sess, base_url, model, key, body)
        return {"t_total_ms": ms, "raw": raw, "usage": usage}

    # ── PART 2: LIVE PAIRED N=64 COMPARISON (C1, C1-rf, C2_flash, C2_flash_stream, C4; keeping completed C2/C2_stream on 3.5-transcribe) ──
    print("[Part 2] Running paired live Gemini API comparison across n=64 utterances...", flush=True)
    rng = random.Random(42)
    conds_p2_run = ["C1", "C1_rf", "C2_flash", "C2_flash_stream", "C4"]
    for idx_u, u in enumerate(utts):
        uid, sc = u["uid"], u["scenario"]
        order = list(conds_p2_run)
        rng.shuffle(order)
        for cond in order:
            ckey = f"p2|{cond}|{uid}"
            if ckey in ckpt:
                continue
            time.sleep(0.15)
            if cond == "C1":
                res = run_c1_audio(direct_base, exp_common.MODEL, uid, sc, sse_rf=False)
                parsed, ok = parse_coach_json(res["raw"])
                rep = parsed.get("reply", "")
                rec = {
                    "ckey": ckey, "uid": uid, "cond": cond, "scenario": sc,
                    "ok": ok, "api_ms": res["t_total_ms"], "stt_ms": 0.0, "coach_ms": res["t_total_ms"],
                    "c0_ready_ms": res["t_total_ms"],
                    "transcript": parsed.get("transcription", ""), "score": parsed.get("score"),
                    "status": parsed.get("status"), "correction": parsed.get("correction", ""),
                    "pronunciation": parsed.get("pronunciation", ""), "reply": rep,
                    "chunk0": first_chunk_apost(rep), "usage": res["usage"],
                    "guardrails": apply_client_guardrails(parsed, parsed.get("transcription", "")),
                }
            elif cond == "C1_rf":
                res = run_c1_audio(direct_base, exp_common.MODEL, uid, sc, sse_rf=True)
                parsed, ok = parse_coach_json(res["raw"])
                rep = parsed.get("reply", "")
                c0 = res["c0_text"] or first_chunk_apost(rep)
                rec = {
                    "ckey": ckey, "uid": uid, "cond": cond, "scenario": sc,
                    "ok": ok, "api_ms": res["t_total_ms"], "stt_ms": 0.0, "coach_ms": res["t_total_ms"],
                    "c0_ready_ms": res["t_c0_ready_ms"], "t_first_tok_ms": res["t_first_ms"],
                    "t_reply_done_ms": res["t_reply_done_ms"],
                    "transcript": parsed.get("transcription", ""), "score": parsed.get("score"),
                    "status": parsed.get("status"), "correction": parsed.get("correction", ""),
                    "pronunciation": parsed.get("pronunciation", ""), "reply": rep,
                    "chunk0": c0, "usage": res["usage"],
                    "guardrails": apply_client_guardrails(parsed, parsed.get("transcription", "")),
                }
            elif cond == "C2_flash":
                stt_ms, tr, u_stt = run_stt_call(direct_base, uid, stt_model=exp_common.MODEL)
                res = run_coach_text(direct_base, exp_common.MODEL, sc, tr, sse_rf=False)
                parsed, ok = parse_coach_json(res["raw"])
                rep = parsed.get("reply", "")
                rec = {
                    "ckey": ckey, "uid": uid, "cond": cond, "scenario": sc,
                    "ok": ok, "api_ms": stt_ms + res["t_total_ms"], "stt_ms": stt_ms, "coach_ms": res["t_total_ms"],
                    "c0_ready_ms": stt_ms + res["t_total_ms"],
                    "transcript": tr, "score": parsed.get("score"),
                    "status": parsed.get("status"), "correction": parsed.get("correction", ""),
                    "pronunciation": parsed.get("pronunciation", ""), "reply": rep,
                    "chunk0": first_chunk_apost(rep), "usage": res["usage"],
                    "guardrails": apply_client_guardrails(parsed, tr),
                }
            elif cond == "C2_flash_stream":
                stt_ms, tr, u_stt = run_stt_call(direct_base, uid, stt_model=exp_common.MODEL)
                res = run_coach_text(direct_base, exp_common.MODEL, sc, tr, sse_rf=True)
                parsed, ok = parse_coach_json(res["raw"])
                rep = parsed.get("reply", "")
                c0 = res["c0_text"] or first_chunk_apost(rep)
                rec = {
                    "ckey": ckey, "uid": uid, "cond": cond, "scenario": sc,
                    "ok": ok, "api_ms": stt_ms + res["t_total_ms"], "stt_ms": stt_ms, "coach_ms": res["t_total_ms"],
                    "c0_ready_ms": stt_ms + res["t_c0_ready_ms"], "t_first_tok_ms": stt_ms + res["t_first_ms"],
                    "t_reply_done_ms": stt_ms + res["t_reply_done_ms"],
                    "transcript": tr, "score": parsed.get("score"),
                    "status": parsed.get("status"), "correction": parsed.get("correction", ""),
                    "pronunciation": parsed.get("pronunciation", ""), "reply": rep,
                    "chunk0": c0, "usage": res["usage"],
                    "guardrails": apply_client_guardrails(parsed, tr),
                }
            elif cond == "C4":
                wrec = whisper_small_map[uid]
                tr = wrec["transcript"]
                res = run_coach_text(direct_base, exp_common.MODEL, sc, tr, sse_rf=False)
                parsed, ok = parse_coach_json(res["raw"])
                rep = parsed.get("reply", "")
                rec = {
                    "ckey": ckey, "uid": uid, "cond": cond, "scenario": sc,
                    "ok": ok, "api_ms": wrec["whisper_ms"] + res["t_total_ms"], "stt_ms": wrec["whisper_ms"], "coach_ms": res["t_total_ms"],
                    "c0_ready_ms": wrec["whisper_ms"] + res["t_total_ms"],
                    "transcript": tr, "score": parsed.get("score"),
                    "status": parsed.get("status"), "correction": parsed.get("correction", ""),
                    "pronunciation": parsed.get("pronunciation", ""), "reply": rep,
                    "chunk0": first_chunk_apost(rep), "usage": res["usage"],
                    "guardrails": apply_client_guardrails(parsed, tr),
                }
            ckpt[ckey] = rec
            append_ckpt(rec)
        if (idx_u + 1) % 8 == 0:
            print(f"  [Part 2] completed {idx_u+1}/{len(utts)} utterances", flush=True)

    # ── PART 3: STRATIFIED SUB-STUDIES (N=24 / N=32) ──
    # Pick 3 per scenario = 24 for History & ProxyNet, and 4 per scenario = 32 for Alt Models
    by_sc = {}
    for u in utts:
        by_sc.setdefault(u["scenario"], []).append(u)
    strat24 = [u for sc_u in by_sc.values() for u in (sc_u[0], sc_u[2], sc_u[5])]  # clean, fragment/error, error mix (8x3=24)
    strat32 = [u for sc_u in by_sc.values() for u in (sc_u[0], sc_u[2], sc_u[4], sc_u[6])]  # 8x4=32

    print("[Part 3a] Running History Depth Ablation (h=8, h=20) on n=24 C1 vs C2...", flush=True)
    for h_depth in [8, 20]:
        for u in strat24:
            uid, sc = u["uid"], u["scenario"]
            hist = build_dummy_history(sc, h_depth)
            ck1 = f"hist_{h_depth}|C1|{uid}"
            if ck1 not in ckpt:
                res = run_c1_audio(direct_base, exp_common.MODEL, uid, sc, history=hist, sse_rf=False)
                parsed, ok = parse_coach_json(res["raw"])
                ckpt[ck1] = {
                    "ckey": ck1, "uid": uid, "cond": "C1", "h": h_depth, "ok": ok,
                    "c0_ready_ms": res["t_total_ms"], "usage": res["usage"],
                    "transcript": parsed.get("transcription", ""), "chunk0": first_chunk_apost(parsed.get("reply", "")),
                }
                append_ckpt(ckpt[ck1])
            ck2 = f"hist_{h_depth}|C2|{uid}"
            if ck2 not in ckpt:
                stt_ms, tr, _ = run_stt_call(direct_base, uid)
                res = run_coach_text(direct_base, exp_common.MODEL, sc, tr, history=hist, sse_rf=False)
                parsed, ok = parse_coach_json(res["raw"])
                ckpt[ck2] = {
                    "ckey": ck2, "uid": uid, "cond": "C2", "h": h_depth, "ok": ok,
                    "c0_ready_ms": stt_ms + res["t_total_ms"], "stt_ms": stt_ms, "coach_ms": res["t_total_ms"],
                    "usage": res["usage"], "transcript": tr, "chunk0": first_chunk_apost(parsed.get("reply", "")),
                }
                append_ckpt(ckpt[ck2])

    print("[Part 3b] Running Shaped HTTP Proxy Network Profiles (Wi-Fi +60ms RTT; LTE +150ms RTT / 1.5 Mbps) on n=24...", flush=True)
    srv, proxy_url = start_shaping_proxy()
    net_profiles = [
        ("wifi_60ms", 60.0, 20_000_000.0),
        ("lte_150ms_1m5", 150.0, 1_500_000.0),
    ]
    for pname, rtt_ms, ul_bps in net_profiles:
        ShapingProxyConfig.rtt_ms = rtt_ms
        ShapingProxyConfig.uplink_bps = ul_bps
        for u in strat24:
            uid, sc = u["uid"], u["scenario"]
            for cond in ["C1", "C2", "C2_stream"]:
                ck = f"net_{pname}|{cond}|{uid}"
                if ck in ckpt:
                    continue
                if cond == "C1":
                    res = run_c1_audio(proxy_url, exp_common.MODEL, uid, sc, sse_rf=False)
                    parsed, ok = parse_coach_json(res["raw"])
                    rec = {"ckey": ck, "net": pname, "cond": cond, "uid": uid, "ok": ok, "c0_ready_ms": res["t_total_ms"], "api_ms": res["t_total_ms"], "chunk0": first_chunk_apost(parsed.get("reply", ""))}
                elif cond == "C2":
                    stt_ms, tr, _ = run_stt_call(proxy_url, uid)
                    res = run_coach_text(proxy_url, exp_common.MODEL, sc, tr, sse_rf=False)
                    parsed, ok = parse_coach_json(res["raw"])
                    rec = {"ckey": ck, "net": pname, "cond": cond, "uid": uid, "ok": ok, "c0_ready_ms": stt_ms + res["t_total_ms"], "api_ms": stt_ms + res["t_total_ms"], "chunk0": first_chunk_apost(parsed.get("reply", ""))}
                else:
                    stt_ms, tr, _ = run_stt_call(proxy_url, uid)
                    res = run_coach_text(proxy_url, exp_common.MODEL, sc, tr, sse_rf=True)
                    parsed, ok = parse_coach_json(res["raw"])
                    c0 = res["c0_text"] or first_chunk_apost(parsed.get("reply", ""))
                    rec = {"ckey": ck, "net": pname, "cond": cond, "uid": uid, "ok": ok, "c0_ready_ms": stt_ms + res["t_c0_ready_ms"], "api_ms": stt_ms + res["t_total_ms"], "chunk0": c0}
                ckpt[ck] = rec
                append_ckpt(rec)
    srv.shutdown()

    print("[Part 3c] Running Alternative Backend Model Generalization (gemini-3.8-flash, gemini-3.5-flash-lite) on n=32...", flush=True)
    for alt_mod in ["gemini-3.8-flash", "gemini-3.5-flash-lite"]:
        for u in strat32:
            uid, sc = u["uid"], u["scenario"]
            ck = f"alt_{alt_mod}|C1|{uid}"
            if ck in ckpt:
                continue
            res = run_c1_audio(direct_base, alt_mod, uid, sc, sse_rf=False)
            parsed, ok = parse_coach_json(res["raw"])
            rec = {
                "ckey": ck, "model": alt_mod, "uid": uid, "scenario": sc, "ok": ok,
                "c0_ready_ms": res["t_total_ms"], "transcript": parsed.get("transcription", ""),
                "score": parsed.get("score"), "status": parsed.get("status"),
                "correction": parsed.get("correction", ""), "reply": parsed.get("reply", ""),
                "chunk0": first_chunk_apost(parsed.get("reply", "")),
                "guardrails": apply_client_guardrails(parsed, parsed.get("transcription", "")),
            }
            ckpt[ck] = rec
            append_ckpt(rec)

    # ── PART 4: SYNTHESIZE ALL UNIQUE CHUNK0 STRINGS CLEANLY VIA KOKORO (4t AND 2t) ──
    all_c0 = sorted({rec.get("chunk0", "") for rec in ckpt.values() if rec.get("chunk0", "")})
    print(f"[Part 4] Synthesizing {len(all_c0)} unique chunk0 replies cleanly via Kokoro 4t & 2t...", flush=True)
    tts_timings = {}
    for nt_label, nt in [("4t", 4), ("2t", 2)]:
        tts = make_kokoro(nt)
        tmap = {}
        for ch in all_c0:
            t0 = time.perf_counter()
            gen = tts.generate(ch, sid=exp_common.VOICE_SID, speed=exp_common.VOICE_SPEED)
            tmap[ch] = ((time.perf_counter() - t0) * 1000.0, (len(gen.samples) / gen.sample_rate) * 1000.0)
        tts_timings[nt_label] = tmap

    # ── AGGREGATE & SAVE ──
    def e2e_ttfa(rec, nt_label="4t"):
        ch = rec.get("chunk0", "")
        tts_ms = tts_timings[nt_label].get(ch, (0.0, 0.0))[0]
        raw_t = rec["c0_ready_ms"] + tts_ms
        # +16.67ms half frame average (~8.3ms) + 9ms client overhead (from instrumentation)
        return raw_t + 9.0 + 8.33

    refs64 = [u["utterance"] for u in utts]

    def summarize_p2_subset(sub_utts, cond_list):
        sub_refs = [u["utterance"] for u in sub_utts]
        res_map = {}
        for cond in cond_list:
            recs = [ckpt[f"p2|{cond}|{u['uid']}"] for u in sub_utts]
            hyps = [r["transcript"] for r in recs]
            scores = [r["score"] for r in recs if isinstance(r.get("score"), (int, float))]
            res_map[cond] = {
                "json_valid": sum(1 for r in recs if r["ok"]),
                "api_ready_ms": bootstrap_ci([r["c0_ready_ms"] for r in recs]),
                "api_full_ms": bootstrap_ci([r["api_ms"] for r in recs]),
                "e2e_4t_ms": bootstrap_ci([e2e_ttfa(r, "4t") for r in recs]),
                "e2e_2t_ms": bootstrap_ci([e2e_ttfa(r, "2t") for r in recs]),
                "raw_wer": wer_stats(sub_refs, hyps, norm_raw),
                "canon_wer": wer_stats(sub_refs, hyps, norm_canon),
                "mean_score": round(statistics.mean(scores), 2) if scores else None,
                "guardrail_r3": sum(1 for r in recs if r["guardrails"]["r3"]),
                "guardrail_r4": sum(1 for r in recs if r["guardrails"]["r4"]),
            }
        return res_map

    def paired_mcnemar_sub(sub_utts, c_a, c_b):
        n11 = n10 = n01 = n00 = 0
        for u in sub_utts:
            r_ref = norm_canon(u["utterance"])
            ha = norm_canon(ckpt[f"p2|{c_a}|{u['uid']}"]["transcript"]) == r_ref
            hb = norm_canon(ckpt[f"p2|{c_b}|{u['uid']}"]["transcript"]) == r_ref
            if ha and hb: n11 += 1
            elif ha and not hb: n10 += 1
            elif not ha and hb: n01 += 1
            else: n00 += 1
        tot_d = n10 + n01
        if tot_d == 0:
            p_ex = 1.0
        else:
            k = min(n10, n01)
            p_ex = min(1.0, 2.0 * sum(math.comb(tot_d, i) for i in range(k + 1)) / (2 ** tot_d))
        return {"n": len(sub_utts), "n11": n11, "n10": n10, "n01": n01, "n00": n00, "exact_p": p_ex}

    matched_transcribe_utts = [
        u for u in utts
        if all(f"p2|{c}|{u['uid']}" in ckpt for c in ["C1", "C1_rf", "C2", "C2_stream", "C2_flash", "C2_flash_stream", "C4"])
    ]
    p2_summary = summarize_p2_subset(utts, conds_p2_run)
    p2_matched_t = summarize_p2_subset(matched_transcribe_utts, ["C1", "C1_rf", "C2", "C2_stream", "C2_flash", "C2_flash_stream", "C4"]) if matched_transcribe_utts else {}

    hist_summary = {}
    for h_d in [0, 8, 20]:
        for cond in ["C1", "C2"]:
            if h_d == 0:
                recs = [ckpt[f"p2|{'C2_flash' if cond=='C2' else 'C1'}|{u['uid']}"] for u in strat24]
            else:
                recs = [ckpt[f"hist_{h_d}|{cond}|{u['uid']}"] for u in strat24]
            ptoks = [r.get("usage", {}).get("promptTokenCount", 0) for r in recs]
            refs24 = [u["utterance"] for u in strat24]
            hyps24 = [r["transcript"] for r in recs]
            hist_summary[f"h{h_d}_{cond}"] = {
                "prompt_tokens_med": statistics.median(ptoks) if ptoks else 0,
                "api_ms": bootstrap_ci([r["c0_ready_ms"] for r in recs]),
                "e2e_4t_ms": bootstrap_ci([e2e_ttfa(r, "4t") for r in recs]),
                "canon_wer": wer_stats(refs24, hyps24, norm_canon),
            }

    net_summary = {}
    for pname in ["lan_direct", "wifi_60ms", "lte_150ms_1m5"]:
        for cond in ["C1", "C2", "C2_stream"]:
            if pname == "lan_direct":
                lk = "C1" if cond == "C1" else ("C2_flash" if cond == "C2" else "C2_flash_stream")
                recs = [ckpt[f"p2|{lk}|{u['uid']}"] for u in strat24]
            else:
                recs = [ckpt[f"net_{pname}|{cond}|{u['uid']}"] for u in strat24]
            net_summary[f"{pname}_{cond}"] = {
                "api_ready_ms": bootstrap_ci([r["c0_ready_ms"] for r in recs]),
                "e2e_4t_ms": bootstrap_ci([e2e_ttfa(r, "4t") for r in recs]),
            }

    alt_summary = {}
    refs32 = [u["utterance"] for u in strat32]
    for mod in [exp_common.MODEL, "gemini-3.8-flash", "gemini-3.5-flash-lite"]:
        if mod == exp_common.MODEL:
            recs = [ckpt[f"p2|C1|{u['uid']}"] for u in strat32]
        else:
            recs = [ckpt[f"alt_{mod}|C1|{u['uid']}"] for u in strat32]
        hyps32 = [r["transcript"] for r in recs]
        scores32 = [r["score"] for r in recs if isinstance(r.get("score"), (int, float))]
        alt_summary[mod] = {
            "json_valid": sum(1 for r in recs if r["ok"]),
            "api_ms": bootstrap_ci([r["c0_ready_ms"] for r in recs]),
            "raw_wer": wer_stats(refs32, hyps32, norm_raw),
            "canon_wer": wer_stats(refs32, hyps32, norm_canon),
            "mean_score": round(statistics.mean(scores32), 2) if scores32 else None,
            "guardrail_r3": sum(1 for r in recs if r["guardrails"]["r3"]),
            "guardrail_r4": sum(1 for r in recs if r["guardrails"]["r4"]),
        }

    out = {
        "part1_whisper_local_64": {
            wn: {
                "latency_ms": bootstrap_ci([r["whisper_ms"] for r in whisper_res[wn]]),
                "raw_wer": wer_stats(refs64, [r["transcript"] for r in whisper_res[wn]], norm_raw),
                "canon_wer": wer_stats(refs64, [r["transcript"] for r in whisper_res[wn]], norm_canon),
            }
            for wn in ["base.en", "small.en"]
        },
        "part2_paired_64": p2_summary,
        "part2_paired_matched_transcribe": {"n": len(matched_transcribe_utts), "conditions": p2_matched_t},
        "part2_mcnemar": {
            "C1_vs_C2_flash_64": paired_mcnemar_sub(utts, "C1", "C2_flash"),
            "C1_vs_C1_rf_64": paired_mcnemar_sub(utts, "C1", "C1_rf"),
            "C1_vs_C4_64": paired_mcnemar_sub(utts, "C1", "C4"),
            "C1_vs_C2_35transcribe": paired_mcnemar_sub(matched_transcribe_utts, "C1", "C2") if matched_transcribe_utts else {},
        },
        "part3a_history_depth_24": hist_summary,
        "part3b_network_profiles_24": net_summary,
        "part3c_model_generalization_32": alt_summary,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
