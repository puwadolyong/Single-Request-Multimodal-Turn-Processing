"""Revision experiment: zero-fallback causal chunking baselines, L_min sweep, Lookahead-1, S+B prebuffer,
adaptive A vs A_post, and 32 disjoint re-chunked long-reply pairs evaluated on Core Ultra 7 (4t idle & 2t pinned).
Uses the exact C# sherpa-onnx 1.13.8 per-chunk caches where chunks match, and freshly synthesizes every
previously unseen chunk via sherpa_onnx.OfflineTts at matching thread count & CPU affinity (ZERO fallback).
"""
import csv
import ctypes
import json
import math
import os
import statistics
import sys
import time
import sherpa_onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_common

KOKORO_DIR = exp_common.KOKORO_DIR
RESULTS_DIR = exp_common.RESULTS
STRAT_PATH = os.path.join(RESULTS_DIR, "stratified_128_rids.json")
OUT_JSON = os.path.join(RESULTS_DIR, "rev_chunking_baselines.json")
FRAME_MS = 1000.0 / 60.0
GAP_THRESH_MS = 20.0


def get_affinity_mask():
    try:
        kernel32 = ctypes.windll.kernel32
        proc = ctypes.c_size_t()
        sys_m = ctypes.c_size_t()
        kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(sys_m))
        return proc.value
    except Exception:
        return None


def set_affinity_mask(mask):
    if mask is None:
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask))
    except Exception:
        pass


def load_csharp_cache(path: str):
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["text"]] = (float(row["synth_ms"]), float(row["audio_ms"]), float(row["resamp_ms"]))
    return out


def make_tts(num_threads: int):
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=os.path.join(KOKORO_DIR, "model.onnx"),
                voices=os.path.join(KOKORO_DIR, "voices.bin"),
                tokens=os.path.join(KOKORO_DIR, "tokens.txt"),
                data_dir=os.path.join(KOKORO_DIR, "espeak-ng-data"),
            ),
            num_threads=num_threads,
        )
    )
    tts = sherpa_onnx.OfflineTts(cfg)
    _ = tts.generate("Warm up sentence for Kokoro.", sid=exp_common.VOICE_SID, speed=exp_common.VOICE_SPEED)
    return tts


def synth_one(tts, text: str):
    t0 = time.perf_counter()
    gen = tts.generate(text, sid=exp_common.VOICE_SID, speed=exp_common.VOICE_SPEED)
    synth_ms = (time.perf_counter() - t0) * 1000.0
    audio_ms = (len(gen.samples) / gen.sample_rate) * 1000.0
    # Resample overhead scaled identically to C# cubic/linear 24k->48k (~0.002 ms per 1s audio -> negligible ~0.01-0.2ms)
    resamp_ms = max(0.01, (len(gen.samples) / 24000.0) * 0.05)
    return (synth_ms, audio_ms, resamp_ms)


def lmin_chunks(sents, thresh: int):
    chunks = []
    cur = ""
    for s in sents:
        cur = s if not cur else f"{cur} {s}"
        if len(cur) >= thresh:
            chunks.append(cur)
            cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def lookahead1_chunks(sents):
    if len(sents) <= 2:
        return [" ".join(sents)]
    return [f"{sents[0]} {sents[1]}"] + list(sents[2:])


def adaptive_post_merge(text: str, r_hat: float = 52.0, alpha: float = 1.3, tau0: float = 220.0, tau1: float = 18.8):
    """Two-pass adaptive sentence chunker A_post (Algorithm 1) with backward-revisiting Pass 2."""
    sents = exp_common.sentences(text)
    if not sents:
        return []
    # Pass 1: forward greedy accumulation
    chunks = [sents[0]]
    for s in sents[1:]:
        if r_hat * len(chunks[-1]) >= alpha * (tau0 + tau1 * len(s)):
            chunks.append(s)
        else:
            chunks[-1] = chunks[-1] + " " + s
    # Pass 2: backward-revisiting lookahead
    i = 0
    while i < len(chunks) - 1:
        if r_hat * len(chunks[i]) < alpha * (tau0 + tau1 * len(chunks[i + 1])):
            chunks[i] = chunks[i] + " " + chunks[i + 1]
            del chunks[i + 1]
            i = max(0, i - 1)
        else:
            i += 1
    return chunks


def causal_online_eval(items, cache, eta: float = 0.25, r_hat0: float = 52.0,
                       alpha: float = 1.3, tau0: float = 220.0, tau1: float = 18.8):
    """Strictly causal Eq. (6) online EMA adaptation evaluated sequentially across turns.

    r_hat is initialized at r_hat0, used prior to scheduling each turn (zero future leakage),
    and updated strictly after each chunk completes synthesis with D_k^obs / len(chunk).
    """
    r_hat = r_hat0
    recs = []
    for _rid, _sc, rep, _ss in items:
        chunks = adaptive_post_merge(rep, r_hat=r_hat, alpha=alpha, tau0=tau0, tau1=tau1)
        st_list = [cache[c] for c in chunks]
        recs.append(eval_schedule(st_list, pre_buffer_ms=0.0, use_60hz=True))
        for ch, (_sm, am, _rm) in zip(chunks, st_list):
            if len(ch) > 0:
                r_hat = (1.0 - eta) * r_hat + eta * (am / len(ch))
    return recs


def eval_schedule(chunk_stats_list, pre_buffer_ms: float = 0.0, use_60hz: bool = True):
    s_clock = 0.0
    p_clock = 0.0
    ttfa_ms = 0.0
    gap_sum = 0.0
    gaps_over_20 = 0
    max_single_gap = 0.0
    for i, (sm, am, rm) in enumerate(chunk_stats_list):
        if use_60hz:
            s_clock += sm + rm
            ready = math.ceil(s_clock / FRAME_MS) * FRAME_MS
            if i == 0:
                ttfa_ms = ready + pre_buffer_ms
                start = ready + pre_buffer_ms
            else:
                desired = max(ready, p_clock)
                start = math.ceil(desired / FRAME_MS) * FRAME_MS
        else:
            s_clock += sm
            if i == 0:
                ttfa_ms = s_clock + pre_buffer_ms
                start = ttfa_ms
            else:
                start = max(s_clock, p_clock)
        if i > 0 and start > p_clock:
            g = start - p_clock
            gap_sum += g
            if g > max_single_gap:
                max_single_gap = g
            if g > GAP_THRESH_MS:
                gaps_over_20 += 1
        p_clock = start + am
    return {
        "ttfa_ms": ttfa_ms,
        "completion_ms": p_clock,
        "gaps_over_20": gaps_over_20,
        "gap_sum_ms": gap_sum,
        "max_gap_ms": max_single_gap,
        "audio_ms": sum(x[1] for x in chunk_stats_list),
    }


def pct(vals, q: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize_metrics(records):
    n = len(records)
    ttfas = sorted(r["ttfa_ms"] for r in records)
    comps = sorted(r["completion_ms"] for r in records)
    aff = sum(1 for r in records if r["gaps_over_20"] > 0)
    tot_gaps = sum(r["gaps_over_20"] for r in records)
    max_single_g = max(r["max_gap_ms"] for r in records)
    aff_sums = [r["gap_sum_ms"] for r in records if r["gaps_over_20"] > 0]
    max_reply_stall = max(aff_sums) if aff_sums else 0.0
    return {
        "n": n,
        "affected_replies": aff,
        "affected_pct": round(aff / n * 100.0, 2),
        "total_gaps_over_20ms": tot_gaps,
        "max_reply_stall_ms": round(max_reply_stall, 2),
        "max_single_gap_ms": round(max_single_g, 2),
        "ttfa_median_ms": round(statistics.median(ttfas), 2),
        "ttfa_mean_ms": round(statistics.mean(ttfas), 2),
        "ttfa_p95_ms": round(pct(ttfas, 0.95), 2),
        "comp_median_ms": round(statistics.median(comps), 2),
        "comp_p95_ms": round(pct(comps, 0.95), 2),
    }


def main():
    with open(STRAT_PATH, "r", encoding="utf-8") as f:
        strat_rids = set(json.load(f)["rids"])

    turns = exp_common.load_turns()
    strat_replies = []
    all_multi = []
    for t in turns:
        rid = f"{t['run']}|{t['turn']:03d}"
        rep = exp_common.reply_of(t.get("raw", ""))
        ss = exp_common.sentences(rep)
        if len(ss) >= 2:
            all_multi.append((rid, t["scenario"], rep, ss))
            if rid in strat_rids:
                strat_replies.append((rid, t["scenario"], rep, ss))

    # Build 32 disjoint three_plus long pairs (0..63 non-overlapping -> 32 pairs, re-chunked across concatenated text)
    three_plus = [x for x in all_multi if len(x[3]) >= 3][:64]
    disjoint_pairs = []
    for i in range(32):
        a = three_plus[2 * i]
        b = three_plus[2 * i + 1]
        comb = f"{a[2]} {b[2]}"
        disjoint_pairs.append((f"{a[0]}+{b[0]}", f"{a[1]}+{b[1]}", comb, exp_common.sentences(comb)))

    print(f"Loaded stratified={len(strat_replies)}, disjoint_pairs={len(disjoint_pairs)}")

    # Build strategy plans per item
    def build_plans(items):
        plans = {}
        for rid, sc, rep, ss in items:
            plans[(rid, "W")] = [rep]
            plans[(rid, "S")] = ss
            plans[(rid, "Lookahead1")] = lookahead1_chunks(ss)
            plans[(rid, "A")] = list(exp_common.adaptive_chunks(rep))
            plans[(rid, "A_post")] = adaptive_post_merge(rep)
            for th in [40, 60, 80, 100, 120]:
                plans[(rid, f"Lmin_{th}")] = lmin_chunks(ss, th)
        return plans

    strat_plans = build_plans(strat_replies)
    pair_plans = build_plans(disjoint_pairs)
    all_unique = set()
    for chs in list(strat_plans.values()) + list(pair_plans.values()):
        all_unique.update(chs)

    orig_mask = get_affinity_mask()
    profiles = [
        ("desktop_4t_idle", 4, orig_mask, os.path.join(RESULTS_DIR, "rev_native_chunk_cache_desktop_4t_idle.csv")),
        ("laptop_2t_affinity4", 2, 0b1111, os.path.join(RESULTS_DIR, "rev_native_chunk_cache_laptop_2t_affinity4.csv")),
    ]

    out = {"stratified_n": len(strat_replies), "disjoint_pairs_n": len(disjoint_pairs), "profiles": {}}

    for pname, nt, aff_mask, cache_path in profiles:
        set_affinity_mask(aff_mask)
        cache = load_csharp_cache(cache_path)
        aug_csv = os.path.join(RESULTS_DIR, f"rev_augmented_chunk_cache_{pname}.csv")
        if os.path.isfile(aug_csv):
            for k_c, v_c in load_csharp_cache(aug_csv).items():
                cache.setdefault(k_c, v_c)
        missing = sorted(c for c in all_unique if c not in cache)
        shared_probe = sorted(c for c in all_unique if c in cache)[:10]
        print(f"[{pname}] total_unique={len(all_unique)}, cached={len(all_unique)-len(missing)}, missing_to_synth={len(missing)}", flush=True)
        parity_diffs = []
        if missing:
            tts = make_tts(nt)
            for c in shared_probe:
                sm_py, _am_py, _ = synth_one(tts, c)
                sm_cs, _am_cs, _ = cache[c]
                parity_diffs.append({"chars": len(c), "cs_synth_ms": round(sm_cs, 2), "py_synth_ms": round(sm_py, 2), "ratio": round(sm_py / max(1e-3, sm_cs), 3)})
            for i, c in enumerate(missing):
                cache[c] = synth_one(tts, c)
                if (i + 1) % 50 == 0:
                    print(f"  [{pname}] synthesized {i+1}/{len(missing)} extra unique chunks", flush=True)
            with open(aug_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["synth_ms", "audio_ms", "resamp_ms", "chars", "text"])
                for txt in sorted(all_unique):
                    sm, am, rm = cache[txt]
                    w.writerow([f"{sm:.3f}", f"{am:.3f}", f"{rm:.3f}", len(txt), txt])

        # Score all strategies on stratified 128 and disjoint 32 pairs
        strat_res = {}
        strat_names = ["W", "S", "S_buf_200", "S_buf_400", "S_buf_600", "S_buf_800", "S_buf_1000",
                       "Lookahead1", "Lmin_40", "Lmin_60", "Lmin_80", "Lmin_100", "Lmin_120",
                       "CausalOnline", "A", "A_post"]
        for st in strat_names:
            if st == "CausalOnline":
                recs = causal_online_eval(strat_replies, cache)
            else:
                recs = []
                for rid, _sc, _rep, _ss in strat_replies:
                    if st.startswith("S_buf_"):
                        pb = float(st.split("_")[-1])
                        chs = strat_plans[(rid, "S")]
                    else:
                        pb = 0.0
                        chs = strat_plans[(rid, st)]
                    st_list = [cache[c] for c in chs]
                    recs.append(eval_schedule(st_list, pre_buffer_ms=pb, use_60hz=True))
            strat_res[st] = summarize_metrics(recs)

        pair_res = {}
        pair_names = ["W", "S", "S_buf_400", "S_buf_800", "Lookahead1", "Lmin_40", "Lmin_60", "Lmin_80", "CausalOnline", "A", "A_post"]
        for st in pair_names:
            if st == "CausalOnline":
                recs = causal_online_eval(disjoint_pairs, cache)
            else:
                recs = []
                for rid, _sc, _comb, _ss in disjoint_pairs:
                    if st.startswith("S_buf_"):
                        pb = float(st.split("_")[-1])
                        chs = pair_plans[(rid, "S")]
                    else:
                        pb = 0.0
                        chs = pair_plans[(rid, st)]
                    st_list = [cache[c] for c in chs]
                    recs.append(eval_schedule(st_list, pre_buffer_ms=pb, use_60hz=True))
            pair_res[st] = summarize_metrics(recs)

        # Evaluate robustness of fixed L_min thresholds vs A_post under faster speaking rate (1.30 vs 1.05) and 1.35x CPU slowdown
        sens_res = {}
        for cond_lbl, synth_scale, speed_scale in [
            ("speed_1.30x", 1.0, 1.30 / 1.05),
            ("cpu_slow_1.35x", 1.35, 1.0),
        ]:
            scaled_cache = {k: (v[0] * synth_scale, v[1] / speed_scale, v[2]) for k, v in cache.items()}
            c_map = {}
            for sub_lbl, items, plans in [("stratified_128", strat_replies, strat_plans), ("disjoint_32", disjoint_pairs, pair_plans)]:
                s_map = {}
                for pol in ("Lmin_40", "Lmin_60", "A_post"):
                    recs = [eval_schedule([scaled_cache[c] for c in plans[(rid, pol)]], pre_buffer_ms=0.0, use_60hz=True)
                            for rid, _sc, _rep, _ss in items]
                    s_map[pol] = summarize_metrics(recs)
                c_map[sub_lbl] = s_map
            sens_res[cond_lbl] = c_map

        out["profiles"][pname] = {
            "threads": nt,
            "synthesized_extra_chunks": len(missing),
            "cache_hit_chunks": len(all_unique) - len(missing),
            "cross_binding_parity_sample": parity_diffs,
            "stratified_128": strat_res,
            "disjoint_32_long_pairs": pair_res,
            "rate_and_slowdown_sensitivity": sens_res,
        }

    # Restore full mask
    set_affinity_mask(orig_mask)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
