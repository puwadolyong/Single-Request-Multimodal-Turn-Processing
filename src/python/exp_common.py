"""Shared pieces for the latency experiments: paths, the on-device voice, the production
sentence chunker (a line-for-line port of SpeechChunker.cs), the scenario library, and the
production coaching prompt / response schema (ports of CoachPrompt.cs and CoachTurnSchema)."""
import json
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ARCHIVE_DATA = os.path.join(ARCHIVE_ROOT, "data")
PROJECT = os.environ.get("CONVIA_PROJECT", "<CONVIA_PROJECT_ROOT>")
KOKORO_DIR = os.environ.get("KOKORO_DIR", os.path.join(PROJECT, r"Assets\StreamingAssets\Kokoro-82M\kokoro-en-v0_19"))
SCENARIOS_CS = os.path.join(PROJECT, r"Assets\Scripts\GeminiAvatar\Scenarios\BuiltInScenarios.cs")
TURN_LOGS = (
    [os.path.join(ARCHIVE_DATA, "turns_run1.jsonl"), os.path.join(ARCHIVE_DATA, "turns_run2.jsonl")]
    if os.path.isfile(os.path.join(ARCHIVE_DATA, "turns_run1.jsonl"))
    else [os.path.join(PROJECT, r"Paper\turns_run1.jsonl"), os.path.join(PROJECT, r"Paper\turns_run2.jsonl")]
)
STUDY_UTTERANCES = os.path.join(PROJECT, r"Assets\StudyData\study_utterances.txt")
# The two logged runs (E1 corpus) were produced on 10 Sep 2026 by the model deployed at the time,
# for typed input; the runner reset the dialogue history every N turns. The application moved to
# MODEL below on 10 Oct 2026, and E2 used MODEL.
CORPUS_MODEL = "gemini-3.5-flash"
CORPUS_HISTORY_RESET = {"run1": 8, "ceiling1": 32}
KEY_FILE = os.environ.get("GEMINI_KEY_FILE", os.path.join(HERE, "gemini.key"))
RESULTS = ARCHIVE_DATA if os.path.isdir(ARCHIVE_DATA) else os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

MODEL = "gemini-3.6-flash"          # GeminiProxy.DefaultModel
TEMPERATURE = 0.9                   # GeminiProxy.Temperature
MAX_OUTPUT_TOKENS = 800
HISTORY_MAX_EXCHANGES = 20          # GeminiChatService._maxHistoryTurns (not sent in E2)
VOICE_SID, VOICE_SPEED, THREADS = 0, 1.05, 4   # GeminiAvatarController defaults
# SpeechChunker.cs: the constants were set during development from timings of seven real replies on a
# Ryzen 5 5600 with 4 threads; speech lasted 52-55 ms per character at speed 1.05 and the lower figure was kept.
CHUNKER_CALIBRATION = {"replies": 7, "spoken_ms_per_char_range": (52, 55)}

# E2 request settings (used by e2_turn_latency.py)
API_VERSION = "v1beta"
TRANSCRIBE_PROMPT = "Transcribe the English speech in this audio word for word. Return only the transcript."
TRANSCRIBE_TEMPERATURE, TRANSCRIBE_MAX_TOKENS = 0.0, 200
CLOUD_TTS_VOICE = "Kore"
LEARNER_VOICES, LEARNER_SPEED = [2, 5, 8], 0.95     # speakers other than the avatar's (0)


# ---------------------------------------------------------------- on-device voice
def load_tts(threads=THREADS):
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=os.path.join(KOKORO_DIR, "model.onnx"),
                voices=os.path.join(KOKORO_DIR, "voices.bin"),
                tokens=os.path.join(KOKORO_DIR, "tokens.txt"),
                data_dir=os.path.join(KOKORO_DIR, "espeak-ng-data"),
            ),
            num_threads=threads,
            provider="cpu",
        ),
        max_num_sentences=1,
    )
    return sherpa_onnx.OfflineTts(cfg)


def synth(tts, text, sid=VOICE_SID, speed=VOICE_SPEED):
    """Returns (samples float32, sample_rate, wall-clock synthesis ms)."""
    t0 = time.perf_counter()
    audio = tts.generate(text, sid=sid, speed=speed)
    ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate, ms


# ---------------------------------------------------------------- SpeechChunker.cs port
SPOKEN_MS_PER_CHAR = 52.0
SYNTH_FIXED_MS = 220.0
SYNTH_MS_PER_CHAR = 18.8
SAFETY = 1.3
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_ABBR = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "mt.", "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.", "no."}


def sentences(text):
    out, pending = [], ""
    for piece in _SENT_END.split(text.strip()):
        if not piece:
            continue
        pending = piece if not pending else pending + " " + piece
        last = pending.rsplit(" ", 1)[-1].lower()
        if last not in _ABBR:
            out.append(pending)
            pending = ""
    if pending:
        out.append(pending)
    return out


def covers(playing, nxt):
    return len(playing) * SPOKEN_MS_PER_CHAR >= (SYNTH_FIXED_MS + len(nxt) * SYNTH_MS_PER_CHAR) * SAFETY


def adaptive_chunks(text):
    s = sentences(text)
    if not s:
        return []
    chunks, cur = [], s[0]
    for nxt in s[1:]:
        if covers(cur, nxt):
            chunks.append(cur)
            cur = nxt
        else:
            cur = cur + " " + nxt
    chunks.append(cur)
    return chunks


def schedule(chunk_stats):
    """chunk_stats: list of (synth_ms, audio_ms) in playback order, synthesised one after
    another on a single worker while earlier chunks play. Returns time to first audio, total
    silent gap after playback started, number of gaps > 20 ms, and completion time (ms)."""
    ready, t = [], 0.0
    for s, _ in chunk_stats:
        t += s
        ready.append(t)
    start = ready[0]
    play_end = start + chunk_stats[0][1]
    gap_total, gaps = 0.0, 0
    for i in range(1, len(chunk_stats)):
        begin = max(ready[i], play_end)
        gap = begin - play_end
        if gap > 20.0:
            gaps += 1
        gap_total += max(0.0, gap)
        play_end = begin + chunk_stats[i][1]
    return start, gap_total, gaps, play_end


# ---------------------------------------------------------------- data
def load_turns():
    turns = []
    for path in TURN_LOGS:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))
    return turns


def reply_of(raw):
    try:
        return json.loads(raw).get("reply", "") or ""
    except Exception:
        m = re.search(r"\[SPOKEN\](.*?)\[/SPOKEN\]", raw, re.S)
        return m.group(1).strip() if m else ""


def load_utterance_classes():
    """(scenario, utterance) -> class A/B/C/D from the scripted study list (B = one grammatical error)."""
    out = {}
    with open(STUDY_UTTERANCES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("|", 2)]
            if len(parts) == 3:
                out[(parts[0], parts[2])] = parts[1]
    return out


def load_scenarios():
    """Persona and objective evaluation rules per scenario, parsed from scenarios.json or BuiltInScenarios.cs."""
    sc_json = os.path.join(ARCHIVE_ROOT, "prompts", "scenarios.json")
    if os.path.isfile(sc_json) and not os.path.isfile(SCENARIOS_CS):
        with open(sc_json, encoding="utf-8") as f:
            return json.load(f)
    src = open(SCENARIOS_CS, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"case PracticeScenario\.(\w+):(.*?)(?=case PracticeScenario\.|\Z)", src, re.S):
        name, body = m.group(1), m.group(2)
        pm = re.search(r'Persona\s*=\s*"((?:[^"\\]|\\.)*)"', body)
        rules = []
        for q in re.finditer(r"new ScenarioQuestDefinition\((.*?)\)\s*[,}]", body, re.S):
            strs = re.findall(r'"((?:[^"\\]|\\.)*)"', q.group(1))
            if len(strs) >= 4:
                rules.append(strs[3])
        if pm:
            out[name] = {"persona": pm.group(1), "rules": rules}
    return out


# ---------------------------------------------------------------- CoachPrompt.cs port
OBJECTIVE_RULES = (
    "QUEST EVALUATION RULES:\n"
    "- List a quest number in questsCompleted only when the user actually accomplished that objective with an understandable sentence.\n"
    "- A bare keyword or fragment does not complete an objective even when the topic matches: \"total price\" is not asking for the total, it is naming a subject. The learner is here to practise saying the whole thing.\n"
    "- An objective also does not count when it makes no sense at this point in the conversation - asking the total before ordering anything, for instance.\n"
    "- Be encouraging and friendly in your reply, but do not hand out objectives that were not earned.\n\n")


def coach_prompt(persona, rules, explain_in_thai=True):
    objectives = ""
    if rules:
        objectives = "SCENARIO QUEST OBJECTIVES TO EVALUATE:\n" + "".join(
            f"- Quest {i + 1}: {r}\n" for i, r in enumerate(rules)) + "\n" + OBJECTIVE_RULES
    thai = ("IMPORTANT: Write all feedback in clear, simple English, with no emojis. The one exception is the tipThai field, which is written in Thai.\n\n"
            if explain_in_thai else
            "IMPORTANT: Write ALL feedback in clear, simple English (Simple English only, no Thai text, no emojis).\n\n")
    prompt_body = os.path.join(HERE, "coach_prompt_body.txt")
    if not os.path.isfile(prompt_body):
        prompt_body = os.path.join(ARCHIVE_ROOT, "prompts", "coach_prompt_body.txt")
    with open(prompt_body, encoding="utf-8") as f:
        body = f.read()
    if explain_in_thai:
        body = body.replace("{TIPTHAI}", "- tipThai: the same explanation as tip, written in natural Thai for a Thai first-language learner.\n")
    else:
        body = body.replace("{TIPTHAI}", "- tipThai: leave this an empty string.\n")
    return persona + "\n\n" + objectives + thai + body


SCHEMA = {
    "type": "object",
    "properties": {
        "transcription": {"type": "string"}, "score": {"type": "integer"},
        "status": {"type": "string", "enum": ["PERFECT", "GOOD", "NEEDS_FIX"]},
        "correction": {"type": "string"}, "tip": {"type": "string"}, "betterPhrasing": {"type": "string"},
        "pronunciation": {"type": "string"}, "tipThai": {"type": "string"}, "reply": {"type": "string"},
        "emotion": {"type": "string", "enum": ["neutral", "happy", "sad", "angry", "surprised", "fear", "disgust", "thinking"]},
        "emotionIntensity": {"type": "integer"},
        "questsCompleted": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["score", "status", "reply", "emotion"],
    "propertyOrdering": ["transcription", "score", "status", "correction", "pronunciation", "tip", "tipThai",
                         "betterPhrasing", "reply", "emotion", "emotionIntensity", "questsCompleted"],
}
VOICE_INSTRUCTION = ("Listen to my spoken English in this audio. Transcribe it, coach it, reply in character, and list any "
                     "quest objectives I accomplished. If you cannot hear any actual speech in the audio, return an empty "
                     "transcription rather than guessing at what I might have said.")
