"""E2 - end-to-end latency of a spoken turn under three decompositions.

Learner utterances are the scripted utterances of the logged live runs, rendered as speech by a
different on-device voice than the avatar's and stored as 16 kHz mono 16-bit WAV, the format the
application records. For each utterance, in randomised condition order:

  C1 single-shot (proposed): one request carrying audio + instruction + schema
  C2 decomposed:             request A transcribes the audio, request B coaches the transcript
                             (same instruction and schema, text input)
  C3 conventional:           C2 followed by request T, which synthesises B's reply in the cloud

Every request goes from this machine straight to the model provider, so all conditions share
one network path. Outputs results/e2_calls.csv (one row per request) and results/e2_turns.csv
(one row per utterance and condition), and saves the audio under results/audio/."""
import base64
import csv
import io
import json
import os
import random
import sys
import time
import wave

import numpy as np
import requests
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E

N_PER_SCENARIO = 8
LEARNER_VOICES = E.LEARNER_VOICES
TTS_MODEL = "gemini-3.8-flash-tts"
STT_MODEL = E.MODEL
URL = "https://generativelanguage.googleapis.com/" + E.API_VERSION + "/models/{m}:generateContent"
KEY = open(E.KEY_FILE, encoding="utf-8").read().strip()
AUDIO_DIR = os.path.join(E.RESULTS, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
rng = random.Random(20261014)

# ------------------------------------------------------------------ utterances
turns = [t for t in E.load_turns() if t["run"].startswith("run1") and t.get("userText")]
by_scen = {}
for t in turns:
    by_scen.setdefault(t["scenario"], []).append(t["userText"])
items = []
for scen in sorted(by_scen):
    pool = sorted(set(by_scen[scen]))
    for u in rng.sample(pool, min(N_PER_SCENARIO, len(pool))):
        items.append((scen, u))
scenarios = E.load_scenarios()
SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:
    items = items[:1]
SUFFIX = "_smoke" if SMOKE else ""


def wav16k(text, sid):
    tts = wav16k.tts
    samples, sr, _ = E.synth(tts, text, sid=sid, speed=E.LEARNER_SPEED)
    y = resample_poly(samples, 2, 3) if sr == 24000 else samples
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
    return buf.getvalue(), len(pcm) / 16000.0


wav16k.tts = E.load_tts(threads=4)

# ------------------------------------------------------------------ requests
calls = []


def post(tag, model, body, uid, cond):
    data = json.dumps(body).encode("utf-8")
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            r = requests.post(URL.format(m=model), data=data, timeout=90,
                              headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
            ms = (time.perf_counter() - t0) * 1000.0
            status = r.status_code
        except requests.RequestException:
            ms, status, r = (time.perf_counter() - t0) * 1000.0, -1, None
        print(f"  {uid} {cond} {tag} attempt={attempt} status={status} ms={ms:.0f}", flush=True)
        if status == 200:
            j = r.json()
            u = j.get("usageMetadata", {})
            calls.append([uid, cond, tag, model, attempt, status, round(ms, 1), len(data), len(r.content),
                          u.get("promptTokenCount", -1), u.get("candidatesTokenCount", -1)])
            return j, ms
        calls.append([uid, cond, tag, model, attempt, status, round(ms, 1), len(data), 0, -1, -1])
        wait = 2.0 * (attempt + 1)
        if status == 429:
            try:
                wait = max(wait, float(r.headers.get("Retry-After", 0)))
            except Exception:
                pass
            wait = max(wait, 15.0 * (attempt + 1))
        time.sleep(wait)
    return None, None


def coach_body(scen, audio_b64=None, text=None):
    sc = scenarios[scen]
    parts = ([{"text": E.VOICE_INSTRUCTION}, {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}}]
             if audio_b64 else [{"text": text}])
    return {
        "systemInstruction": {"parts": [{"text": E.coach_prompt(sc["persona"], sc["rules"])}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": E.TEMPERATURE, "maxOutputTokens": E.MAX_OUTPUT_TOKENS,
                             "thinkingConfig": {"thinkingBudget": 0},
                             "responseMimeType": "application/json", "responseSchema": E.SCHEMA},
    }


def stt_body(audio_b64):
    return {
        "contents": [{"role": "user", "parts": [
            {"text": E.TRANSCRIBE_PROMPT},
            {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}}]}],
        "generationConfig": {"temperature": E.TRANSCRIBE_TEMPERATURE, "maxOutputTokens": E.TRANSCRIBE_MAX_TOKENS, "thinkingConfig": {"thinkingBudget": 0}},
    }


def tts_body(text):
    return {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"responseModalities": ["AUDIO"],
                             "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": E.CLOUD_TTS_VOICE}}}},
    }


def text_of(j):
    try:
        return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
    except Exception:
        return ""


def coach_fields(j):
    try:
        o = json.loads(text_of(j))
        return o.get("reply", ""), o.get("transcription", ""), o.get("score", ""), o.get("pronunciation", "")
    except Exception:
        return "", "", "", ""


def reply_from(j):
    return coach_fields(j)[0]


def save():
    with open(os.path.join(E.RESULTS, "e2_calls" + SUFFIX + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "condition", "request", "model", "attempt", "status", "ms", "request_bytes",
                    "response_bytes", "prompt_tokens", "output_tokens"])
        w.writerows(calls)
    with open(os.path.join(E.RESULTS, "e2_turns" + SUFFIX + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "scenario", "utterance", "audio_s", "condition", "ok", "ms_first", "ms_second",
                    "ms_tts", "network_ms", "reply", "transcript", "score", "pronunciation"])
        w.writerows(turn_rows)


# ------------------------------------------------------------------ run
turn_rows = []
for k, (scen, utt) in enumerate(items):
    uid = f"u{k:02d}"
    wav, dur = wav16k(utt, LEARNER_VOICES[k % len(LEARNER_VOICES)])
    open(os.path.join(AUDIO_DIR, uid + ".wav"), "wb").write(wav)
    b64 = base64.b64encode(wav).decode("ascii")
    conds = ["C1", "C2", "C3"]
    rng.shuffle(conds)
    for cond in conds:
        if cond == "C1":
            j, ms = post("single", E.MODEL, coach_body(scen, audio_b64=b64), uid, cond)
            rep, tr, sc, pr = coach_fields(j) if j else ("", "", "", "")
            turn_rows.append([uid, scen, utt, round(dur, 3), cond, ms is not None, round(ms or 0, 1), 0, 0,
                              round(ms or 0, 1), rep, tr, sc, pr])
        else:
            ja, ma = post("stt", STT_MODEL, stt_body(b64), uid, cond)
            transcript = text_of(ja).strip() if ja else ""
            jb, mb = post("coach", E.MODEL, coach_body(scen, text=transcript), uid, cond) if ja else (None, None)
            reply, _, sc, pr = coach_fields(jb) if jb else ("", "", "", "")
            mt = 0.0
            ok = ja is not None and jb is not None
            if cond == "C3" and ok and reply:
                jt, mt = post("tts", TTS_MODEL, tts_body(reply), uid, cond)
                ok = jt is not None
                mt = mt or 0.0
            total = (ma or 0) + (mb or 0) + (mt or 0)
            turn_rows.append([uid, scen, utt, round(dur, 3), cond, ok, round(ma or 0, 1), round(mb or 0, 1),
                              round(mt or 0, 1), round(total, 1), reply, transcript, sc, pr])
        time.sleep(1.0)
    print(f"{k + 1}/{len(items)} utterances", flush=True)
    save()

save()
print("done")
