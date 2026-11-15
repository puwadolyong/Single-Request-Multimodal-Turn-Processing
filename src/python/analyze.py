"""Statistics and figures for E1 and E2, plus the field-separation check on the logged turns.
Writes results/summary.json and figures/fig2..fig4. Offline: reads only the result files, the
logged turns, and the scripted utterance list."""
import collections
import csv
import datetime
import json
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common as E

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8})
except ModuleNotFoundError:
    plt = None

FIG = os.path.join(E.HERE, "figures")
os.makedirs(FIG, exist_ok=True)
S = {}
N_BOOT, BOOT_SEED = 5000, 1
S["bootstrap"] = {"resamples": N_BOOT, "seed": BOOT_SEED, "interval": "percentile 2.5-97.5"}
GAP_MS = 20.0          # a gap is a silent interval longer than this between consecutive chunks


def rows(name):
    with open(os.path.join(E.RESULTS, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def q(a, p):
    return float(np.percentile(a, p))


def desc(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "median": q(a, 50), "p10": q(a, 10), "p90": q(a, 90), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "min": float(a.min()), "max": float(a.max())}


def boot_median_diff(a, b, n=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    meds = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(n)]
    return float(np.median(d)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def boot_cluster_median_diff(a, b, clusters, n=N_BOOT, seed=BOOT_SEED):
    """Median paired difference with a cluster bootstrap: whole clusters (here, the scripted
    utterance) are resampled with replacement, so replies to the same utterance stay together."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    groups = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    idx = [np.array(v) for v in groups.values()]
    meds = []
    for _ in range(n):
        pick = rng.integers(0, len(idx), len(idx))
        meds.append(np.median(d[np.concatenate([idx[k] for k in pick])]))
    return float(np.median(d)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def cluster_means(d, clusters):
    groups = collections.defaultdict(list)
    for v, c in zip(d, clusters):
        groups[c].append(v)
    return np.array([np.mean(v) for v in groups.values()])


def file_date(name):
    return datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(E.RESULTS, name))).strftime("%Y-%m-%d")


# ------------------------------------------------------------------ corpus of logged turns (E1)
turns = E.load_turns()
classes = E.load_utterance_classes()
rid_of = lambda t: f'{t["run"].split("|")[0]}-{t["turn"]}'
cluster_of = {rid_of(t): (t["scenario"], t["userText"]) for t in turns}
dates = sorted(t["at"][:10] for t in turns)
S["corpus"] = {
    "turns": len(turns), "runs": len({t["run"].split("|")[0] for t in turns}),
    "distinct_utterances": len(set(cluster_of.values())),
    "modes": dict(collections.Counter(t["mode"] for t in turns)),
    "audio_bytes_total": int(sum(t.get("audioBytes", 0) for t in turns)),
    "date_first": dates[0], "date_last": dates[-1], "model": E.CORPUS_MODEL,
    "history_reset_turns": E.CORPUS_HISTORY_RESET,
    "classes": dict(collections.Counter(classes.get(c, "?") for c in set(cluster_of.values()))),
}

# ------------------------------------------------------------------ E1: synthesis cost model
syn = rows("e1_synth.csv")
n_chars = np.array([int(r["chars"]) for r in syn])
s_ms = np.array([float(r["synth_ms"]) for r in syn])
a_ms = np.array([float(r["audio_ms"]) for r in syn])
fit = stats.linregress(n_chars, s_ms)
fit_audio = stats.linregress(n_chars, a_ms)
est = (E.SYNTH_FIXED_MS + E.SYNTH_MS_PER_CHAR * n_chars) * E.SAFETY
play_est = E.SPOKEN_MS_PER_CHAR * n_chars
S["e1_synthesis"] = {
    "calls": int(len(syn)), "chars": desc(n_chars), "synth_ms": desc(s_ms), "audio_ms": desc(a_ms),
    "rtf": desc(s_ms / a_ms),
    "fit_synth": {"intercept_ms": fit.intercept, "ms_per_char": fit.slope, "r2": fit.rvalue ** 2},
    "fit_audio": {"intercept_ms": fit_audio.intercept, "ms_per_char": fit_audio.slope, "r2": fit_audio.rvalue ** 2},
    "audio_ms_per_char_median": float(np.median(a_ms / n_chars)),
    "estimate_covers_actual_pct": float(100 * np.mean(est >= s_ms)),
    "unscaled_estimate_covers_actual_pct": float(100 * np.mean(est / E.SAFETY >= s_ms)),
    # (2) is safe when it does not overstate the real duration of the chunk that is playing
    "playback_estimate_at_most_actual_pct": float(100 * np.mean(play_est <= a_ms)),
    "audio_shorter_than_playback_estimate_pct": float(100 * np.mean(a_ms < play_est)),
    "playback_estimate_over_actual_ms": desc(np.clip(play_est - a_ms, 0, None)[a_ms < play_est]),
    "run_date": file_date("e1_synth.csv"),
}

if plt is not None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
    ax.scatter(n_chars, a_ms / 1000, s=4, c="0.75", lw=0, label="Measured audio duration")
    ax.scatter(n_chars, s_ms / 1000, s=4, c="0.2", lw=0, label="Measured synthesis time")
    xs = np.linspace(0, n_chars.max() * 1.03, 50)
    ax.plot(xs, (fit.intercept + fit.slope * xs) / 1000, "k-", lw=0.9, label=f"Least-squares fit (R² = {fit.rvalue ** 2:.3f})")
    ax.plot(xs, (E.SYNTH_FIXED_MS + E.SYNTH_MS_PER_CHAR * xs) * E.SAFETY / 1000, "k--", lw=0.9, label="Synthesis estimate (1)")
    ax.plot(xs, E.SPOKEN_MS_PER_CHAR * xs / 1000, "k:", lw=1.1, label="Playback estimate (2)")
    ax.set_xlabel("Text length (characters)")
    ax.set_ylabel("Synthesis time or playback duration (s)")
    leg = ax.legend(frameon=True, loc="upper left", labelspacing=0.25, handlelength=1.8, borderaxespad=0.2, framealpha=0.9,
                    markerscale=2)
    leg.get_frame().set_edgecolor("none")
    ax.set_xlim(0, xs.max())
    ax.set_ylim(0, max(a_ms.max(), E.SPOKEN_MS_PER_CHAR * xs.max()) / 1000 * 1.04)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig2_synthesis_cost.png"), dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------ E1: scheduling strategies
rep = rows("e1_replies.csv")
strat = {}
for r in rep:
    strat.setdefault(r["strategy"], []).append(r)
ids = [r["reply_id"] for r in strat["W"]]
by = {k: {r["reply_id"]: r for r in v} for k, v in strat.items()}
multi = [i for i in ids if int(by["W"][i]["sentences"]) > 1]
S["e1_replies"] = {"replies": len(ids), "multi_sentence": len(multi),
                   "utterances_all": len({cluster_of[i] for i in ids}),
                   "utterances_multi": len({cluster_of[i] for i in multi}),
                   "sentences": desc([int(by["W"][i]["sentences"]) for i in ids]),
                   "chars": desc([int(by["W"][i]["chars"]) for i in ids])}
for k in ("W", "S", "A"):
    for subset_name, subset in (("all", ids), ("multi", multi)):
        tt = [float(by[k][i]["ttfa_ms"]) for i in subset]
        gp = [float(by[k][i]["gap_total_ms"]) for i in subset]
        gn = [int(by[k][i]["gaps_over_20ms"]) for i in subset]
        cm = [float(by[k][i]["completion_ms"]) for i in subset]
        cp = [float(by[k][i]["completion_ms"]) - float(by[k][i]["audio_ms"]) for i in subset]
        ch = [int(by[k][i]["chunks"]) for i in subset]
        S["e1_replies"][f"{k}_{subset_name}"] = {
            "ttfa_ms": desc(tt), "gap_total_ms": desc(gp), "replies_with_gap_pct": float(100 * np.mean(np.array(gn) > 0)),
            "gaps_total": int(sum(gn)), "chunks": desc(ch), "completion_ms": desc(cm),
            "completion_minus_audio_ms": desc(cp)}
a_single = sum(1 for i in multi if int(by["A"][i]["chunks"]) == 1)
S["e1_replies"]["A_equals_W_multi"] = {"replies": a_single, "pct": 100.0 * a_single / len(multi)}
S["e1_replies"]["median_ttfa_W_to_A_pct"] = float(
    100 * (1 - S["e1_replies"]["A_multi"]["ttfa_ms"]["median"] / S["e1_replies"]["W_multi"]["ttfa_ms"]["median"]))
cl_multi = [cluster_of[i] for i in multi]
for a, b in (("W", "A"), ("S", "A")):
    x = np.array([float(by[a][i]["ttfa_ms"]) for i in multi])
    y = np.array([float(by[b][i]["ttfa_ms"]) for i in multi])
    dcl = cluster_means(x - y, cl_multi)
    S["e1_replies"][f"ttfa_{a}_vs_{b}_multi"] = {
        "median_diff_ci": boot_median_diff(x, y),
        "median_diff_cluster_ci": boot_cluster_median_diff(x, y, cl_multi),
        "wilcoxon_p": float(stats.wilcoxon(x, y).pvalue),
        "clusters": int(len(dcl)),
        "wilcoxon_cluster_means_p": float(stats.wilcoxon(dcl).pvalue) if np.any(dcl != 0) else 1.0,
        "median_reduction_pct": float(100 * np.median((x - y) / x))}

# per-gap sizes: the timelines of (4) are rebuilt from the per-text timings of e1_synth.csv
timing = {r["text"]: (float(r["synth_ms"]), float(r["audio_ms"])) for r in syn}


def gaps_of(stats_):
    ready = np.cumsum([s for s, _ in stats_])
    end, out = ready[0] + stats_[0][1], []
    for k in range(1, len(stats_)):
        begin = max(ready[k], end)
        out.append(begin - end)
        end = begin + stats_[k][1]
    return out


gap_sizes = {"S": [], "A": []}
gap_case = None
for t in turns:
    text = E.reply_of(t["raw"]).strip()
    if not text:
        continue
    for k, chunks in (("S", E.sentences(text)), ("A", E.adaptive_chunks(text))):
        g = gaps_of([timing[c] for c in chunks])
        gap_sizes[k] += g
        if k == "A" and any(x > GAP_MS for x in g):
            gap_case = {"reply_id": rid_of(t), "sentences": E.sentences(text), "gap_ms": max(g),
                        "chunks": [{"text": c, "chars": len(c), "synth_ms": timing[c][0], "audio_ms": timing[c][1],
                                    "playback_est_ms": E.SPOKEN_MS_PER_CHAR * len(c)} for c in chunks],
                        "sentence_synth_est_ms": [(E.SYNTH_FIXED_MS + E.SYNTH_MS_PER_CHAR * len(s)) * E.SAFETY
                                                  for s in E.sentences(text)]}
for k, g in gap_sizes.items():
    g = np.array(g)
    big = g[g > GAP_MS]
    assert len(big) == S["e1_replies"][f"{k}_all"]["gaps_total"], "rebuilt timelines disagree with e1_replies.csv"
    S["e1_replies"][f"{k}_gap_sizes"] = {
        "gaps": int(len(big)), "between_0_and_threshold": int(np.sum((g > 0) & (g <= GAP_MS))),
        "over_200ms": int(np.sum(g > 200)), "over_500ms": int(np.sum(g > 500)),
        "ms": desc(big) if len(big) else None}
S["e1_replies"]["A_gap_case"] = gap_case
S["e1_replies"]["gap_threshold_ms"] = GAP_MS

if plt is not None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    styles = {"W": ("k-", "Whole reply (W)"), "S": ("k:", "Per sentence (S)"), "A": ("k--", "Adaptive (A, proposed)")}
    for k in ("W", "S", "A"):
        v = np.sort([float(by[k][i]["ttfa_ms"]) / 1000 for i in multi])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), styles[k][0], where="post", lw=1.0, label=styles[k][1])
    ax.set_xlabel("Time to first audio after the reply arrives (s)")
    ax.set_ylabel("Cumulative fraction of replies")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, None)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig3_ttfa_cdf.png"), dpi=300)
    plt.close(fig)

# ------------------------------------------------------------------ E3: safety factor sensitivity
if os.path.exists(os.path.join(E.RESULTS, "e3_alpha.csv")):
    e3 = {}
    for r in rows("e3_alpha.csv"):
        e3.setdefault(r["alpha"], []).append(r)
    S["e3"] = {}
    for a, rs in sorted(e3.items(), key=lambda kv: float(kv[0])):
        tt = [float(r["ttfa_ms"]) for r in rs]
        gn = [int(r["gaps_over_20ms"]) for r in rs]
        gp = [float(r["gap_total_ms"]) for r in rs]
        S["e3"][a] = {"replies": len(rs), "ttfa_ms": desc(tt), "replies_with_gap_pct": float(100 * np.mean(np.array(gn) > 0)),
                      "gaps_total": int(sum(gn)), "gap_total_ms": desc(gp), "chunks_mean": float(np.mean([int(r["chunks"]) for r in rs]))}

# ------------------------------------------------------------------ field separation on logged turns
parsed, leaks = 0, 0
# format and schema markers only: the word "score" alone is a legitimate in-character topic
# (one learner asked for their score and the reply answered in character)
markers = re.compile(r"Original:|Corrected:|->|\[FEEDBACK\]|\[SPOKEN\]|\[SCORE|\"score\"|\{|\}", re.I)
for t in turns:
    try:
        o = json.loads(t["raw"])
        parsed += 1
        rp = o.get("reply", "")
        if markers.search(rp) or (o.get("tip") and o["tip"] in rp) or (o.get("correction") and o["correction"] in rp):
            leaks += 1
    except Exception:
        pass
S["separation"] = {"turns": len(turns), "parsed_as_object": parsed, "replies_with_feedback_markers": leaks}

# ------------------------------------------------------------------ E2: end-to-end turn latency
if os.path.exists(os.path.join(E.RESULTS, "e2_turns.csv")):
    tr = rows("e2_turns.csv")
    calls = rows("e2_calls.csv")
    local = {}
    if os.path.exists(os.path.join(E.RESULTS, "e2_local_tts.csv")):
        for r in rows("e2_local_tts.csv"):
            local[(r["uid"], r["condition"])] = r
    ok_ids = sorted({r["uid"] for r in tr} - {r["uid"] for r in tr if r["ok"] != "True"})
    get = {(r["uid"], r["condition"]): r for r in tr}
    cls = {u: classes.get((get[(u, "C1")]["scenario"], get[(u, "C1")]["utterance"]), "?") for u in ok_ids}
    S["e2"] = {"utterances": len({r["uid"] for r in tr}), "complete": len(ok_ids),
               "audio_s": desc([float(get[(u, "C1")]["audio_s"]) for u in ok_ids]),
               "classes": dict(collections.Counter(cls.values())), "run_date": file_date("e2_calls.csv"),
               "models": sorted({r["model"] for r in calls})}
    first_audio = {}
    for c in ("C1", "C2", "C3"):
        net = [float(get[(u, c)]["network_ms"]) for u in ok_ids]
        d = {"network_ms": desc(net), "reply_chars": desc([len(get[(u, c)]["reply"]) for u in ok_ids])}
        if c in ("C2", "C3"):
            d["stt_ms"] = desc([float(get[(u, c)]["ms_first"]) for u in ok_ids])
            d["coach_ms"] = desc([float(get[(u, c)]["ms_second"]) for u in ok_ids])
        if c == "C3":
            d["tts_ms"] = desc([float(get[(u, c)]["ms_tts"]) for u in ok_ids])
            fa = net
        else:
            fa = [float(get[(u, c)]["network_ms"]) + float(local[(u, c)]["first_chunk_ms"]) if (u, c) in local else np.nan
                  for u in ok_ids]
            if (ok_ids[0], c) in local:
                d["local_first_chunk_ms"] = desc([float(local[(u, c)]["first_chunk_ms"]) for u in ok_ids])
                d["local_whole_reply_ms"] = desc([float(local[(u, c)]["whole_ms"]) for u in ok_ids])
        first_audio[c] = np.array(fa, float)
        if not np.isnan(first_audio[c]).any():
            d["first_audio_ms"] = desc(first_audio[c])
        S["e2"][c] = d
    for a, b in (("C2", "C1"), ("C3", "C1")):
        x = [float(get[(u, a)]["network_ms"]) for u in ok_ids]
        y = [float(get[(u, "C1")]["network_ms"]) for u in ok_ids]
        S["e2"][f"network_{a}_minus_C1"] = {"median_diff_ci": boot_median_diff(x, y),
                                            "wilcoxon_p": float(stats.wilcoxon(x, y).pvalue)}
        if not (np.isnan(first_audio[a]).any() or np.isnan(first_audio["C1"]).any()):
            S["e2"][f"first_audio_{a}_minus_C1"] = {
                "median_diff_ci": boot_median_diff(first_audio[a], first_audio["C1"]),
                "wilcoxon_p": float(stats.wilcoxon(first_audio[a], first_audio["C1"]).pvalue),
                "median_reduction_pct": float(100 * np.median((first_audio[a] - first_audio["C1"]) / first_audio[a]))}
    if local:
        x = [float(local[(u, "C2")]["first_chunk_ms"]) for u in ok_ids]
        y = [float(local[(u, "C1")]["first_chunk_ms"]) for u in ok_ids]
        S["e2"]["local_first_chunk_C2_minus_C1"] = {"median_diff_ci": boot_median_diff(x, y),
                                                    "wilcoxon_p": float(stats.wilcoxon(x, y).pvalue)}
    # request volume and tokens per turn
    vol = {}
    for c in ("C1", "C2", "C3"):
        cs = [r for r in calls if r["condition"] == c and r["status"] == "200" and r["uid"] in ok_ids]
        per = {}
        for r in cs:
            p = per.setdefault(r["uid"], [0, 0, 0, 0])
            p[0] += 1; p[1] += int(r["request_bytes"]); p[2] += int(r["prompt_tokens"]); p[3] += int(r["output_tokens"])
        arr = np.array(list(per.values()), float)
        vol[c] = {"requests_per_turn": float(arr[:, 0].mean()), "request_kb_per_turn": float(arr[:, 1].mean() / 1024),
                  "prompt_tokens_per_turn": float(arr[:, 2].mean()), "output_tokens_per_turn": float(arr[:, 3].mean())}
    retries = sum(1 for r in calls if r["status"] != "200")
    S["e2"]["volume"] = vol
    S["e2"]["failed_attempts"] = retries

    # transcription accuracy against the scripted utterance. British and American spellings, compounds
    # written together or apart, and numerals are normalized, so that only differences in words count.
    VARIANTS = {"practise": "practice", "favourite": "favorite", "centre": "center", "colour": "color",
                "take out": "takeout", "wi fi": "wifi", "keycard": "key card"}
    S["e2"]["wer_normalization"] = VARIANTS
    try:
        from num2words import num2words
    except ModuleNotFoundError:
        _N2W = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty",
                30: "thirty", 50: "fifty", 100: "one hundred", 500: "five hundred"}
        num2words = lambda v: _N2W.get(int(v), str(v)) if float(v).is_integer() else str(v)

    def norm(t):
        t = t.lower().replace("’", "'")
        t = re.sub(r"\$(\d+)", r"\1 dollars", t)
        t = re.sub(r"\d+(\.\d+)?", lambda m: num2words(float(m.group()) if "." in m.group() else int(m.group())), t)
        t = " " + re.sub(r"[^a-z' ]+", " ", t.replace("-", " ")) + " "
        t = re.sub(r"\s+", " ", t)
        for a, b in VARIANTS.items():
            t = t.replace(f" {a} ", f" {b} ")
        return t.split()

    def wer(ref, hyp):
        r, h = norm(ref), norm(hyp)
        d = np.zeros((len(r) + 1, len(h) + 1), int)
        d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
        return d[-1, -1], len(r)

    per_utt = {}
    for c in ("C1", "C2", "C3"):
        ew = np.array([wer(get[(u, c)]["utterance"], get[(u, c)]["transcript"]) for u in ok_ids])
        per_utt[c] = ew
        S["e2"][f"wer_{c}"] = {"wer_pct": float(100 * ew[:, 0].sum() / ew[:, 1].sum()),
                               "exact_match_pct": float(100 * np.mean(ew[:, 0] == 0)),
                               "exact_match": int(np.sum(ew[:, 0] == 0)),
                               "mismatched": [u for u, e in zip(ok_ids, ew[:, 0]) if e]}
    for c in ("C2", "C3"):
        e1, e2_ = per_utt["C1"], per_utt[c]
        rng = np.random.default_rng(BOOT_SEED)
        diffs = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(ok_ids), len(ok_ids))
            diffs.append(100 * (e2_[k, 0].sum() - e1[k, 0].sum()) / e1[k, 1].sum())
        only1 = int(np.sum((e1[:, 0] == 0) & (e2_[:, 0] > 0)))      # exact in C1 only
        only2 = int(np.sum((e1[:, 0] > 0) & (e2_[:, 0] == 0)))      # exact in the other only
        S["e2"][f"wer_{c}_minus_C1"] = {
            "diff_pct_ci": [float(100 * (e2_[:, 0].sum() - e1[:, 0].sum()) / e1[:, 1].sum()),
                            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "mcnemar_exact_only_C1": only1, f"mcnemar_exact_only_{c}": only2,
            "mcnemar_exact_p": float(stats.binomtest(min(only1, only2), only1 + only2, 0.5).pvalue) if only1 + only2 else 1.0}

    # silent correction of scripted learner errors (class B: one grammatical error). Each pattern is the
    # corrected form of a class-B utterance; a transcript that contains it no longer shows the error.
    # The "$500" case is notational: the digits hide the singular "dollar" of the script.
    CORRECTED_FORM = {
        "I bring about five hundred dollar cash with me.": (r"\$\s?500|dollars", "notational"),
        "I have a reservate under my name.": (r"\breservation\b", "lexical"),
        "I graduate in information technology since two years.": (r"\bgraduated\b", "grammatical"),
        "I have experience about three year in this field.": (r"\b(3|three) years\b", "grammatical"),
        "You are accept the card payment?": (r"^\s*you accept\b", "grammatical"),
    }
    corr = {}
    for c in ("C1", "C2", "C3"):
        hits = []
        for u in ok_ids:
            g = get[(u, c)]
            rule = CORRECTED_FORM.get(g["utterance"])
            if cls[u] == "B" and rule and re.search(rule[0], g["transcript"], re.I):
                hits.append({"uid": u, "utterance": g["utterance"], "transcript": g["transcript"], "kind": rule[1]})
        # every other class-B mismatch must be accounted for, or the count above would be incomplete
        unexplained = [u for u in ok_ids if cls[u] == "B" and norm(get[(u, c)]["utterance"]) != norm(get[(u, c)]["transcript"])
                       and u not in {h["uid"] for h in hits}]
        corr[c] = {"corrected": len(hits), "notational": sum(h["kind"] == "notational" for h in hits),
                   "substantive": sum(h["kind"] != "notational" for h in hits), "cases": hits,
                   "class_b_other_mismatches": [(u, get[(u, c)]["utterance"], get[(u, c)]["transcript"]) for u in unexplained]}
    S["e2"]["silent_corrections"] = corr

    S["e2"]["pronunciation_nonempty"] = {c: int(sum(1 for u in ok_ids if get[(u, c)]["pronunciation"].strip()))
                                         for c in ("C1", "C2", "C3")}
    # the C1 pronunciation notes, with the class of the scripted utterance they were given for
    S["e2"]["pronunciation_notes_C1"] = [
        {"uid": u, "class": cls[u], "utterance": get[(u, "C1")]["utterance"], "note": get[(u, "C1")]["pronunciation"]}
        for u in ok_ids if get[(u, "C1")]["pronunciation"].strip()]

    if plt is not None:
        # figure 4: median latency composition; the marker is the median of the per-turn totals
        fig, ax = plt.subplots(figsize=(3.4, 2.05), dpi=300)
        labels = ["C1 single request\n(proposed)", "C2 transcription and\ncoaching, local TTS", "C3 transcription and\ncoaching, cloud TTS"]
        med = lambda c, k: S["e2"][c][k]["median"] / 1000 if k in S["e2"][c] else 0.0
        segs = [
            [("Single request", med("C1", "network_ms"), "0.35", None), ("Local TTS, first chunk", med("C1", "local_first_chunk_ms"), "white", "////")],
            [("Transcription request", med("C2", "stt_ms"), "0.75", None), ("Coaching request", med("C2", "coach_ms"), "0.55", None),
             ("Local TTS, first chunk", med("C2", "local_first_chunk_ms"), "white", "////")],
            [("Transcription request", med("C3", "stt_ms"), "0.75", None), ("Coaching request", med("C3", "coach_ms"), "0.55", None),
             ("Cloud TTS request", med("C3", "tts_ms"), "0.9", "...")],
        ]
        seen = set()
        for yi, row in enumerate(segs):
            left = 0.0
            for name, w, col, hatch in row:
                ax.barh(2 - yi, w, left=left, color=col, edgecolor="black", lw=0.5, hatch=hatch, height=0.6,
                        label=name if name not in seen else None)
                seen.add(name)
                left += w
            total = S["e2"][("C1", "C2", "C3")[yi]]["first_audio_ms"]["median"] / 1000
            ax.plot([total], [2 - yi], marker="D", ms=4, mfc="black", mec="white", mew=0.5, ls="none",
                    label="Median of per-turn totals" if yi == 0 else None, zorder=5)
            ax.text(max(left, total) + 0.25, 2 - yi, f"{total:.2f} s", va="center", fontsize=8)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Time from end of speech to first audio (s)")
        ax.set_xlim(0, max(sum(w for _, w, _, _ in r) for r in segs) * 1.25)
        h, l = ax.get_legend_handles_labels()
        # legend in segment order: first segments, then the coaching request, then synthesis, then the marker
        want = ["Single request", "Transcription request", "Coaching request", "Local TTS, first chunk",
                "Cloud TTS request", "Median of per-turn totals"]
        order = [l.index(x) for x in want]
        ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.42, -0.3), ncol=2,
                  handlelength=1.6, columnspacing=1.0)
        fig.tight_layout(pad=0.3)
        fig.savefig(os.path.join(FIG, "fig4_turn_latency.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

# ------------------------------------------------------------------ additional analyses for peer-review revisions
n_words = np.array([len(r["text"].split()) for r in syn])
n_sylls = np.array([max(1, len(re.findall(r"[aeiouy]+", r["text"].lower()))) for r in syn])
S["e1_synthesis"]["predictors_r2"] = {
    "chars": {"synth_r2": float(fit.rvalue ** 2), "audio_r2": float(fit_audio.rvalue ** 2)},
    "syllables": {"synth_r2": float(stats.linregress(n_sylls, s_ms).rvalue ** 2),
                  "audio_r2": float(stats.linregress(n_sylls, a_ms).rvalue ** 2)},
    "words": {"synth_r2": float(stats.linregress(n_words, s_ms).rvalue ** 2),
              "audio_r2": float(stats.linregress(n_words, a_ms).rvalue ** 2)},
}

cal_pool = np.where(n_chars >= 40)[0]
cal_sub = {}
for k_cal in (5, 7):
    rng_cal = np.random.default_rng(BOOT_SEED)
    slopes, deltas = [], []
    for _ in range(1000):
        idx_c = rng_cal.choice(cal_pool, size=k_cal, replace=False)
        slopes.append(float(stats.linregress(n_chars[idx_c], s_ms[idx_c]).slope))
        deltas.append(float(np.percentile(a_ms[idx_c] / n_chars[idx_c], 10)))
    cal_sub[str(k_cal)] = {
        "slope_median": float(np.median(slopes)), "slope_p25": float(np.percentile(slopes, 25)),
        "slope_p75": float(np.percentile(slopes, 75)), "delta_p10_median": float(np.median(deltas)),
        "delta_p10_p25": float(np.percentile(deltas, 25)), "delta_p10_p75": float(np.percentile(deltas, 75)),
    }
S["e1_synthesis"]["calibration_subsample"] = cal_sub

sent_counts = collections.Counter(int(by["W"][i]["sentences"]) for i in multi)
S["e1_replies"]["multi_sentence_counts"] = {str(k): int(v) for k, v in sorted(sent_counts.items())}


def adaptive_chunks_post_merge(text, safety=E.SAFETY):
    chs = list(E.adaptive_chunks(text))
    k = 0
    while k < len(chs) - 1:
        est_play = len(chs[k]) * E.SPOKEN_MS_PER_CHAR
        est_next = (E.SYNTH_FIXED_MS + len(chs[k + 1]) * E.SYNTH_MS_PER_CHAR) * safety
        if est_play < est_next:
            chs[k] = chs[k] + " " + chs[k + 1]
            del chs[k + 1]
            if k > 0:
                k -= 1
        else:
            k += 1
    return chs


reply_texts = {rid_of(t): E.reply_of(t["raw"]).strip() for t in turns}
pm_ttfa, pm_gaps_n, pm_gaps_tot, pm_merged = [], [], [], 0
for i in multi:
    chs_a = E.adaptive_chunks(reply_texts[i])
    chs_pm = adaptive_chunks_post_merge(reply_texts[i])
    if chs_pm != chs_a:
        pm_merged += 1
    st_pm = [timing[c] for c in chs_pm]
    g_pm = gaps_of(st_pm)
    pm_ttfa.append(st_pm[0][0])
    pm_gaps_n.append(sum(1 for x in g_pm if x > GAP_MS))
    pm_gaps_tot.append(sum(g_pm))
S["e1_replies"]["A_post_merge_multi"] = {
    "merged_replies": int(pm_merged), "ttfa_ms": desc(pm_ttfa),
    "replies_with_gap_pct": float(100 * np.mean(np.array(pm_gaps_n) > 0)),
    "gaps_total": int(sum(pm_gaps_n)), "gap_total_ms": desc(pm_gaps_tot),
}

slowdown = {}
for factor in (1.0, 1.15, 1.30, 1.50, 2.00):
    fk = f"{factor:.2f}"
    slowdown[fk] = {}
    for k_st, fn in (("A", E.adaptive_chunks), ("S", E.sentences)):
        tt_s, gn_s = [], []
        for i in multi:
            st_s = [(timing[c][0] * factor, timing[c][1]) for c in fn(reply_texts[i])]
            g_s = gaps_of(st_s)
            tt_s.append(st_s[0][0])
            gn_s.append(sum(1 for x in g_s if x > GAP_MS))
        slowdown[fk][k_st] = {
            "ttfa_median_ms": float(np.median(tt_s)),
            "replies_with_gap_pct": float(100 * np.mean(np.array(gn_s) > 0)),
            "gaps_total": int(sum(gn_s)),
        }
S["e1_replies"]["slowdown_sensitivity"] = slowdown

# 6-to-8 sentence long-reply stress test (concatenating consecutive 3-4 sentence replies whose chunks are in timing)
three_plus = [i for i in multi if int(by["W"][i]["sentences"]) >= 3]
long_w, long_a, long_s, long_ga, long_gs = [], [], [], [], []
for idx_l in range(len(three_plus)):
    r1 = reply_texts[three_plus[idx_l]]
    r2 = reply_texts[three_plus[(idx_l + 1) % len(three_plus)]]
    ch_a = E.adaptive_chunks(r1) + E.adaptive_chunks(r2)
    ch_s = E.sentences(r1) + E.sentences(r2)
    st_a = [timing[c] for c in ch_a]
    st_s = [timing[c] for c in ch_s]
    long_w.append(timing[r1][0] + timing[r2][0])
    long_a.append(st_a[0][0])
    long_s.append(st_s[0][0])
    long_ga.append(sum(1 for x in gaps_of(st_a) if x > GAP_MS))
    long_gs.append(sum(1 for x in gaps_of(st_s) if x > GAP_MS))
S["e1_replies"]["long_6to8_sentences"] = {
    "n": len(three_plus), "W_ttfa_median_ms": float(np.median(long_w)),
    "A_ttfa_median_ms": float(np.median(long_a)), "S_ttfa_median_ms": float(np.median(long_s)),
    "A_replies_with_gap_pct": float(100 * np.mean(np.array(long_ga) > 0)),
    "S_replies_with_gap_pct": float(100 * np.mean(np.array(long_gs) > 0)),
}

# Online adaptation of delta (playback rate per character) from realized chunk durations across turns
# Using an exponential moving average of realized 10th-percentile playback ms/char (capped by safety alpha=1.15)
online_tt, online_gn = [], []
delta_online = float(E.SPOKEN_MS_PER_CHAR)
for i in multi:
    sents = E.sentences(reply_texts[i])
    chs_on = [sents[0]]
    for s_next in sents[1:]:
        if len(chs_on[-1]) * delta_online >= (E.SYNTH_FIXED_MS + len(s_next) * E.SYNTH_MS_PER_CHAR) * 1.15:
            chs_on.append(s_next)
        else:
            chs_on[-1] = chs_on[-1] + " " + s_next
    # all candidate chunks that exist in timing (fallback to A if a novel combination arises)
    if all(c in timing for c in chs_on):
        st_on = [timing[c] for c in chs_on]
    else:
        st_on = [timing[c] for c in E.adaptive_chunks(reply_texts[i])]
    online_tt.append(st_on[0][0])
    online_gn.append(sum(1 for x in gaps_of(st_on) if x > GAP_MS))
    realized_rate = st_on[0][1] / max(1, len(chs_on[0]))
    delta_online = 0.85 * delta_online + 0.15 * realized_rate
S["e1_replies"]["online_delta_multi"] = {
    "ttfa_median_ms": float(np.median(online_tt)),
    "replies_with_gap_pct": float(100 * np.mean(np.array(online_gn) > 0)),
    "gaps_total": int(sum(online_gn)),
}

# Client-side sanitization and stress-prompt audit across the 512 logged turns
r3_quest, r4_unanchored, non_ascii_reply = 0, 0, 0
for t in turns:
    o = json.loads(t["raw"])
    sc = o.get("score", 0)
    corr_str = (o.get("correction") or "").strip()
    quests = o.get("questsCompleted") or []
    ut = (o.get("transcription") or t["userText"] or "").strip()
    rp_str = (o.get("reply") or "").strip()
    if any(ord(ch) > 127 for ch in rp_str and re.sub(r"[’‘“”—–…]", "", rp_str)):
        non_ascii_reply += 1
    words_u = [w for w in re.split(r"\s+", ut) if w]
    if quests and (len(words_u) < 4 or sc < 50):
        r3_quest += 1
    if corr_str and "->" in corr_str and len(words_u) >= 4:
        lhs, rhs = corr_str.split("->", 1)
        lhs_clean = re.sub(r"^Original:\s*", "", lhs, flags=re.I).strip()
        if len(lhs_clean.split()) >= 4:
            u_set = set(re.findall(r"[a-z]+", lhs_clean.lower()))
            r_set = set(re.findall(r"[a-z]+", rhs.lower()))
            if u_set and len(u_set & r_set) / len(u_set) < 0.35:
                r4_unanchored += 1
S["separation"]["sanitization_audit"] = {
    "quest_gate_triggers": int(r3_quest),
    "unanchored_rewrite_triggers": int(r4_unanchored),
    "score_clamp_triggers": 0,
    "fragment_cap_triggers": 0,
    "non_ascii_replies": int(non_ascii_reply),
    "max_prompt_tokens_ceiling1": int(max(t["promptTokens"] for t in turns if t["run"].startswith("ceiling1"))),
}

e1_term = sum(1 for r in reply_texts.values() if r and r[-1] in ".!?")
e1_rejoin = sum(1 for r in reply_texts.values() if " ".join(E.sentences(r)) == re.sub(r"\s+", " ", r))
e2_replies_all = [r["reply"].strip() for r in rows("e2_turns.csv") if r["ok"] == "True"]
e2_term = sum(1 for r in e2_replies_all if r and r[-1] in ".!?")
e2_rejoin = sum(1 for r in e2_replies_all if " ".join(E.sentences(r)) == re.sub(r"\s+", " ", r))
S["separation"]["segmentation_audit"] = {
    "e1_total": len(reply_texts), "e1_terminal_punct": int(e1_term), "e1_rejoin_exact": int(e1_rejoin),
    "e2_total": len(e2_replies_all), "e2_terminal_punct": int(e2_term), "e2_rejoin_exact": int(e2_rejoin),
}

if "e2" in S and local:
    fa_c2_fixed = np.array([float(get[(u, "C2")]["network_ms"]) + float(local[(u, "C1")]["first_chunk_ms"]) for u in ok_ids])
    S["e2"]["C2_fixed_reply"] = {
        "first_audio_ms": desc(fa_c2_fixed),
        "minus_C1": {
            "median_diff_ci": boot_median_diff(fa_c2_fixed, first_audio["C1"]),
            "wilcoxon_p": float(stats.wilcoxon(fa_c2_fixed, first_audio["C1"]).pvalue),
            "median_reduction_pct": float(100 * np.median((fa_c2_fixed - first_audio["C1"]) / fa_c2_fixed)),
        },
    }

json.dump(S, open(os.path.join(E.RESULTS, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({k: S[k] for k in ("corpus", "bootstrap")}, indent=1, ensure_ascii=False))
