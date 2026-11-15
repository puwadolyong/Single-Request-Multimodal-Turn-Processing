"""Deterministic offline verification script for all manuscript tables and metrics.

Uses only the Python standard library to read archived CSV/JSON logs in ./data/
and recompute every table, statistical test, and audit reported in the revised
IJIES manuscript (Paper ID 20267603).
"""
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def percentile(vals, q):
    """Linear-interpolation percentile matching numpy default."""
    s = sorted(vals)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def med(vals):
    return statistics.median(vals) if vals else float("nan")


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy)
    return intercept, slope, r2


def exact_mcnemar_p(n10, n01):
    """Two-sided exact binomial test p-value for discordant pairs (n10, n01)."""
    k = n10 + n01
    if k == 0:
        return 1.0
    m = min(n10, n01)
    p_tail = sum(math.comb(k, i) for i in range(m + 1)) * (0.5 ** k)
    return min(1.0, 2.0 * p_tail)


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    off = json.loads((DATA / "rev_offline_reconciled.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 1. Experiment 1: Synthesis Fit & Strategy Comparison (CSV + JSON)
    # ------------------------------------------------------------------
    header("1. EXPERIMENT 1: SYNTHESIS FIT (n=1,450) & SCHEDULING (n=484 / N=512)")
    with (DATA / "e1_synth.csv").open(encoding="utf-8") as f:
        synth_rows = list(csv.DictReader(f))
    assert len(synth_rows) == 1450
    chars = [float(r["chars"]) for r in synth_rows]
    sms = [float(r["synth_ms"]) for r in synth_rows]
    ams = [float(r["audio_ms"]) for r in synth_rows]
    rtfs = [s / a for s, a in zip(sms, ams)]
    b0, b1, r2_s = ols(chars, sms)
    _, _, r2_a = ols(chars, ams)
    eq1_cov = sum(1 for c, s in zip(chars, sms) if 1.3 * (220.0 + 18.8 * c) > s) / len(chars)
    eq1_unscaled = sum(1 for c, s in zip(chars, sms) if (220.0 + 18.8 * c) > s) / len(chars)
    eq2_shorter = sum(1 for c, a in zip(chars, ams) if a < 52.0 * c) / len(chars)
    eq2_diffs = [52.0 * c - a for c, a in zip(chars, ams) if a < 52.0 * c]
    print(f"Synth OLS: T(n) = {b0:.1f} + {b1:.1f}*n (R^2={r2_s:.3f}); Audio R^2={r2_a:.3f}, med={med([a/c for a,c in zip(ams,chars)]):.1f} ms/ch")
    print(f"RTF med={med(rtfs):.2f} (P10={percentile(rtfs,0.10):.2f}, P90={percentile(rtfs,0.90):.2f})")
    print(f"Eq(1) exceeds measured: {eq1_cov*100:.1f}% (unscaled a=1: {eq1_unscaled*100:.1f}%)")
    print(f"Eq(2) shorter than 52*n: {eq2_shorter*100:.1f}% (median shortfall {med(eq2_diffs):.0f} ms)")

    with (DATA / "e1_replies.csv").open(encoding="utf-8") as f:
        rep_rows = list(csv.DictReader(f))
    by_strat_all = defaultdict(list)
    by_strat_multi = defaultdict(list)
    sent_counts = Counter()
    for r in rep_rows:
        st = r["strategy"]
        by_strat_all[st].append(r)
        if int(r["sentences"]) > 1:
            by_strat_multi[st].append(r)
        if st == "W":
            sent_counts[int(r["sentences"])] += 1

    n_total = len(by_strat_all["W"])
    n_multi = len(by_strat_multi["W"])
    assert n_total == 512 and n_multi == 484
    print(f"Logged replies N={n_total}, multi-sentence n={n_multi} (sentence dist: {dict(sorted(sent_counts.items()))})")
    print(f"{'Strat':<6} {'Chunks':>7} {'MedTTFA(s)':>11} {'P90(s)':>8} {'Aff/484(%)':>11} {'Aff/512(%)':>11} {'Gaps':>6} {'Gap(ms)':>8} {'Done(s)':>8}")
    for st in ("W", "S", "A"):
        rows_m = by_strat_multi[st]
        ttfa = [float(r["ttfa_ms"]) / 1000.0 for r in rows_m]
        comp = [float(r["completion_ms"]) / 1000.0 for r in rows_m]
        ch = statistics.mean(int(r["chunks"]) for r in rows_m)
        aff_m = sum(1 for r in rows_m if int(r["gaps_over_20ms"]) > 0)
        gaps_m = sum(int(r["gaps_over_20ms"]) for r in rows_m)
        gap_mean = statistics.mean(float(r["gap_total_ms"]) for r in rows_m)
        print(f"{st:<6} {ch:7.2f} {med(ttfa):11.2f} {percentile(ttfa,0.90):8.2f} "
              f"{aff_m:3d} ({aff_m/n_multi*100:4.1f}%) "
              f"{aff_m:3d} ({aff_m/n_total*100:4.1f}%) "
              f"{gaps_m:6d} {gap_mean:8.1f} {med(comp):8.2f}")

    with (DATA / "e3_alpha.csv").open(encoding="utf-8") as f:
        e3_rows = list(csv.DictReader(f))
    print("\nSafety factor alpha sensitivity on multi-sentence n=484 (from e3_alpha.csv):")
    for a_str in ("1.0", "1.15", "1.3", "1.5"):
        sub = [r for r in e3_rows if r["alpha"] == a_str]
        assert len(sub) == 484
        aff = sum(1 for r in sub if int(r["gaps_over_20ms"]) > 0)
        gaps = sum(int(r["gaps_over_20ms"]) for r in sub)
        tt = med([float(r["ttfa_ms"]) / 1000.0 for r in sub])
        ch = statistics.mean(int(r["chunks"]) for r in sub)
        print(f"  alpha={a_str:<4}: chunks={ch:.2f}, medTTFA={tt:.2f} s, aff={aff} ({aff/n_multi*100:.1f}%), gaps={gaps}")

    print("\nPre-buffer (S+B), L_min Ryzen cached & Long-pair comparison (rev_offline_reconciled.json):")
    for k, v in off["e1_ryzen_prebuffer_baselines"].items():
        print(f"  {k:<12}: medTTFA={v['ttfa_ms']['median']/1000:.2f} s, P90={v['ttfa_ms']['p90']/1000:.2f} s, "
              f"aff={v['affected_replies']} ({v['pct_multi_484']:.2f}% of 484 / {v['pct_all_512']:.2f}% of 512), "
              f"done={v['completion_ms']['median']/1000:.2f} s")
    for k, v in off["e1_ryzen_lmin_cached_subset"].items():
        print(f"  {k:<12} (cached n={v['matched_in_ryzen_cache']}/484): medTTFA={v['ttfa_ms']['median']/1000:.2f} s, "
              f"aff={v['affected_replies_cached']} ({v['pct_cached']:.1f}%), done={v['completion_ms']['median']/1000:.2f} s")
    for k, v in off["e1_long_reply_breakdown"].items():
        print(f"  {k}: n={v['n_pairs']}, W_ttfa={v['W_ttfa_ms']['median']/1000:.2f} s, "
              f"A_ttfa={v['A_ttfa_ms']['median']/1000:.2f} s (aff={v['A_affected_replies']}/{v['n_pairs']}={v['A_affected_pct']:.2f}%), "
              f"S_ttfa={v['S_ttfa_ms']['median']/1000:.2f} s (aff={v['S_affected_replies']}/{v['n_pairs']}={v['S_affected_pct']:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Client Instrumentation, 1x Real-Time 60 Hz & Native C# Baselines
    # ------------------------------------------------------------------
    header("2. CLIENT INSTRUMENTATION, REAL-TIME 60HZ VALIDATION & N=128 HARDWARE BASELINES")
    inst = json.loads((DATA / "rev_instrument_e2e.json").read_text(encoding="utf-8"))
    cold_loads = inst["coldModelLoadMs"]
    cold_inf = inst["coldFirstInferMs"]
    warm = inst["stages"]
    print(f"waveOutAvailable={inst['waveOutAvailable']}, cold starts n={len(cold_loads)}: "
          f"modelLoad med={med(cold_loads):.1f} ms ({min(cold_loads):.1f}-{max(cold_loads):.1f}), "
          f"firstInfer med={med(cold_inf):.1f} ms ({min(cold_inf):.1f}-{max(cold_inf):.1f})")
    for field in ("micValidateTimingMs", "wavAndBase64EncodeMs", "proxyLocalLoopbackMs",
                  "clientParseAndChunkMs", "resample48kAndDriverEnqueueMs",
                  "totalNonNetworkClientMs", "warmChunk0SynthMs"):
        vals = [w[field] for w in warm]
        print(f"  Warm {field:<30} (n={len(vals)}): med={med(vals):8.3f} ms, P95={percentile(vals,0.95):8.3f} ms, max={max(vals):8.3f} ms")

    rt60 = json.loads((DATA / "rev_realtime_60hz_validation.json").read_text(encoding="utf-8"))
    print("\n1x Wall-clock 60 Hz waveOut validation (n=24):")
    for st in ("S", "A_post"):
        sub = [r for r in rt60 if r["strategy"] == st]
        aff = sum(1 for r in sub if r["rtGapsOver20Ms"] > 0)
        gaps = sum(r["rtGapsOver20Ms"] for r in sub)
        onset = med([r["rtOnsetMs"] for r in sub])
        comp = med([r["rtCompletionMs"] for r in sub])
        maxg = max(r["rtTotalGapMs"] for r in sub)
        print(f"  {st:<7} (n={len(sub)}): onset={onset:7.1f} ms, comp={comp:7.1f} ms, aff={aff}/{len(sub)} ({aff/len(sub)*100:.1f}%), gaps={gaps}, max_gap_sum={maxg:.1f} ms")

    print("\nNative C# 60 Hz WinMM playback N=128 recomputed from rev_native_playback_128.csv:")
    with (DATA / "rev_native_playback_128.csv").open(encoding="utf-8-sig") as f:
        pb_rows = list(csv.DictReader(f))
    by_prof = defaultdict(list)
    for r in pb_rows:
        by_prof[(r["profile"], r["strategy"])].append(r)
    for prof in ("desktop_4t_idle", "desktop_4t_render_load", "laptop_2t_affinity4"):
        print(f"  [{prof}]")
        for st in ("W", "S", "A", "A_post"):
            sub = by_prof[(prof, st)]
            assert len(sub) == 128
            ttfa_r = [float(x["ttfa_recon_ms"]) for x in sub]
            ttfa_a = [float(x["ttfa_actual_ms"]) for x in sub]
            comp_a = [float(x["completion_actual_ms"]) for x in sub]
            aff_a = sum(1 for x in sub if int(x["gaps_actual"]) > 0)
            gaps_a = sum(int(x["gaps_actual"]) for x in sub)
            maxg_a = max(float(x["gap_actual_ms"]) for x in sub if int(x["gaps_actual"]) > 0) if gaps_a > 0 else 0.0
            print(f"    {st:<6}: n={len(sub)}, TTFA_recon={med(ttfa_r):7.1f}, TTFA_act={med(ttfa_a):7.1f} "
                  f"(P95={percentile(ttfa_a,0.95):7.1f}), Comp={med(comp_a):7.1f}, Aff={aff_a:2d}/128 ({aff_a/128*100:5.2f}%), "
                  f"gaps={gaps_a:2d}, maxStall={maxg_a:6.1f} ms")

    cb = json.loads((DATA / "rev_chunking_baselines.json").read_text(encoding="utf-8"))
    print("\nZero-fallback baseline sweep summary (from rev_chunking_baselines.json):")
    for prof in cb["profiles"]:
        print(f"  [{prof}] (stratified n=128)")
        for pol in ("W", "S", "S_buf_200", "S_buf_400", "S_buf_600", "S_buf_800", "S_buf_1000",
                    "Lookahead1", "Lmin_40", "Lmin_60", "Lmin_80", "CausalOnline", "A", "A_post"):
            d = cb["profiles"][prof]["stratified_128"].get(pol)
            if not d:
                continue
            print(f"    {pol:<13}: TTFA={d['ttfa_median_ms']:7.1f} (P95={d['ttfa_p95_ms']:7.1f}), "
                  f"Comp={d['comp_median_ms']:7.1f}, Aff={d['affected_replies']:2d}/128 ({d['affected_pct']:5.2f}%), "
                  f"maxG={d['max_single_gap_ms']:6.1f}")
        print(f"  [{prof}] (disjoint n=32 long pairs)")
        for pol in ("W", "S", "S_buf_400", "S_buf_800", "Lookahead1", "Lmin_80", "CausalOnline", "A", "A_post"):
            d = cb["profiles"][prof]["disjoint_32_long_pairs"].get(pol)
            if not d:
                continue
            print(f"    {pol:<13}: TTFA={d['ttfa_median_ms']:7.1f}, Aff={d['affected_replies']:2d}/32 ({d['affected_pct']:5.2f}%), maxG={d['max_single_gap_ms']:6.1f}")

    # ------------------------------------------------------------------
    # 3. Experiment 2 Latency, WER, McNemar & Utterance Inventory
    # ------------------------------------------------------------------
    header("3. EXPERIMENT 2 TURN LATENCY (N=64), WER & MCNEMAR RECONCILIATION")
    with (DATA / "e2_turns.csv").open(encoding="utf-8") as f:
        e2_rows = list(csv.DictReader(f))
    with (DATA / "e2_local_tts.csv").open(encoding="utf-8") as f:
        tts_rows = list(csv.DictReader(f))
    tts_map = {(r["uid"], r["condition"]): r for r in tts_rows}
    for cond in ("C1", "C2", "C3"):
        sub = [r for r in e2_rows if r["condition"] == cond]
        rem = [float(r["network_ms"]) / 1000.0 for r in sub]
        fa = []
        for r in sub:
            net_ms = float(r["network_ms"])
            if cond in ("C1", "C2"):
                t_fc = float(tts_map[(r["uid"], cond)]["first_chunk_ms"])
                fa.append((net_ms + t_fc) / 1000.0)
            else:
                fa.append(net_ms / 1000.0)
        print(f"  {cond}: n={len(sub)}, Remote med={med(rem):.2f} s, FirstAudio med={med(fa):.2f} s (P90={percentile(fa,0.90):.2f} s)")

    c1_map = {r["uid"]: r for r in e2_rows if r["condition"] == "C1"}
    c2_map = {r["uid"]: r for r in e2_rows if r["condition"] == "C2"}
    c2_fixed = [(float(c2_map[u]["network_ms"]) + float(tts_map[(u, "C1")]["first_chunk_ms"])) / 1000.0 for u in sorted(c1_map)]
    c1_fa = [(float(c1_map[u]["network_ms"]) + float(tts_map[(u, "C1")]["first_chunk_ms"])) / 1000.0 for u in sorted(c1_map)]
    diffs = [b - a for a, b in zip(c1_fa, c2_fixed)]
    print(f"  Fixed-reply C2 med={med(c2_fixed):.2f} s, paired median diff={med(diffs):.2f} s")

    wer_rec = off["e2_transcription_reconciliation"]
    for mode in ("raw", "canonical"):
        sec = wer_rec[mode]
        print(f"\n  [{mode.upper()} WER]")
        for c, m in sec["conditions"].items():
            print(f"    {c}: exact={m['exact_matches']}/{m['total']} ({m['exact_pct']:.2f}%), "
                  f"S={m['substitutions']}, D={m['deletions']}, I={m['insertions']}, WER={m['wer_pct']:.2f}%")
        for pair, mc in sec["mcnemar_2x2"].items():
            p_chk = exact_mcnemar_p(mc["n10_only_first_exact"], mc["n01_only_second_exact"])
            assert abs(p_chk - mc["exact_binomial_p"]) < 1e-12
            print(f"    McNemar {pair}: (11={mc['n11_both_exact']}, 10={mc['n10_only_first_exact']}, "
                  f"01={mc['n01_only_second_exact']}, 00={mc['n00_neither_exact']}) p={p_chk:.5f}")
    print(f"\n  Utterance-by-condition diff inventory count: {len(wer_rec['utterance_diff_inventory'])} utterances with any raw or canonical mismatch")

    # ------------------------------------------------------------------
    # 4. Live API Ablations (tb0 & default thinking) + SpeechOcean762 + Guardrails
    # ------------------------------------------------------------------
    header("4. LIVE API ABLATIONS (tb0 & Default Thinking), PORTABILITY & CAUSAL FIELD ORDER")
    tb0 = json.loads((DATA / "rev_live_api_ablations_tb0.json").read_text(encoding="utf-8"))
    print("Dedicated Whisper baseline (n=64 CPU int8 4t):")
    for mname, d in tb0["part1_whisper_local_64"].items():
        print(f"  {mname:<10}: p50={d['latency_ms']['p50']:6.1f} ms (P95={d['latency_ms']['p95']:6.1f}), "
              f"raw={d['raw_wer']['exact_pct']:5.2f}% (WER {d['raw_wer']['wer_pct']:4.2f}%), "
              f"canon={d['canon_wer']['exact_pct']:5.2f}% (WER {d['canon_wer']['wer_pct']:4.2f}%)")

    print("\nZero-thinking (thinkingBudget=0, maxOutputTokens=2048) paired n=64:")
    for c, d in tb0["part2_paired_64"].items():
        print(f"  {c:<16}: json={d['json_valid']}/64, apiReady={d['api_ready_ms']['p50']:6.1f} (P95={d['api_ready_ms']['p95']:6.1f}), "
              f"e2e4t={d['e2e_4t_ms']['p50']:6.1f} (P95={d['e2e_4t_ms']['p95']:6.1f}), e2e2t={d['e2e_2t_ms']['p50']:6.1f}, "
              f"canonEx={d['canon_wer']['exact_pct']:5.2f}% (WER {d['canon_wer']['wer_pct']:4.2f}%), score={d['mean_score']:.2f}")
    for pair, mc in tb0["part2_mcnemar"].items():
        if not mc:
            continue
        p_chk = exact_mcnemar_p(mc["n10"], mc["n01"])
        assert abs(p_chk - mc["exact_p"]) < 1e-12
        print(f"  McNemar {pair}: (11={mc['n11']}, 10={mc['n10']}, 01={mc['n01']}, 00={mc['n00']}) p={p_chk:.5f}")

    print("\nHistory depth & Network profiles (n=24, tb0):")
    for k, d in tb0["part3a_history_depth_24"].items():
        print(f"  {k:<8}: tok={d['prompt_tokens_med']:6.1f}, api={d['api_ms']['p50']:6.1f} (P95={d['api_ms']['p95']:6.1f}), "
              f"e2e4t={d['e2e_4t_ms']['p50']:6.1f}, canonEx={d['canon_wer']['exact_pct']:5.2f}%")
    for k, d in tb0["part3b_network_profiles_24"].items():
        print(f"  {k:<22}: apiReady={d['api_ready_ms']['p50']:6.1f} (P95={d['api_ready_ms']['p95']:6.1f}), "
              f"e2e4t={d['e2e_4t_ms']['p50']:6.1f} (P95={d['e2e_4t_ms']['p95']:6.1f})")

    print("\nCross-model portability (n=32, tb0):")
    for k, d in tb0["part3c_model_generalization_32"].items():
        print(f"  {k:<24}: json={d['json_valid']}/32, api={d['api_ms']['p50']:6.1f} (P95={d['api_ms']['p95']:6.1f}), "
              f"canonEx={d['canon_wer']['exact_pct']:5.2f}% (WER {d['canon_wer']['wer_pct']:5.2f}%), score={d['mean_score']:.2f}")

    def_th = json.loads((DATA / "rev_live_api_ablations.json").read_text(encoding="utf-8"))
    sub51 = def_th["part2_paired_matched_transcribe"]
    print(f"\nLegacy 800-token default-thinking audit (matched n={sub51['n']} w/ gemini-3.5-transcribe):")
    for c, d in sub51["conditions"].items():
        print(f"  {c:<16}: json={d['json_valid']}/{sub51['n']}, apiReady={d['api_ready_ms']['p50']:6.1f}, "
              f"e2e4t={d['e2e_4t_ms']['p50']:6.1f}, canonEx={d['canon_wer']['exact_pct']:5.2f}% (WER {d['canon_wer']['wer_pct']:4.2f}%)")

    # ------------------------------------------------------------------
    # 5. Human L2 Speech (SpeechOcean762, N=100) & C# Guardrail Audit
    # ------------------------------------------------------------------
    header("5. HUMAN L2 SPEECH (SpeechOcean762, N=100) & C# GUARDRAIL AUDIT (N=512)")
    so = json.loads((DATA / "rev_speechocean762.json").read_text(encoding="utf-8"))
    print(f"Stratified sample n={so['sample_n']} (adults={so['adults_n']}, youth={so['youth_n']}, "
          f"accuracy<=6={so['accuracy_le_6_n']}, word_misp={so['with_word_mispronunciation_n']})")
    for k, w in so["asr_comparison"].items():
        jstr = f", json={w['json_valid']}/{w['raw']['n']}" if "json_valid" in w else ""
        print(f"  {k:<28}: n={w['raw']['n']}{jstr}, lat_med={w['latency_med_ms']:6.1f} ms, "
              f"raw_ex={w['raw']['exact']}/100 ({w['raw']['exact_pct']:.1f}%), raw_WER={w['raw']['wer_pct']:.2f}%, "
              f"canon_WER={w['canon']['wer_pct']:.2f}%")
    corr = so["human_score_alignment"]
    print(f"  Score vs Human Experts (Spearman): acc={corr['spearman_vs_accuracy']:.4f}, "
          f"fluency={corr['spearman_vs_fluency']:.4f}, total={corr['spearman_vs_total']:.4f}")
    for st, vals in corr["strata"].items():
        print(f"  Stratum {st:<14}: n={vals['n']}, convia mean={vals['convia_score_mean']:.2f}, med={vals['convia_score_med']:.1f}")
    for tau, metrics in so["pronunciation_abstention_vs_acc_le6"].items():
        print(f"  Abstain (acc<=6) tau={tau:<8}: TP={metrics['tp']:2d}, FP={metrics['fp']:2d}, FN={metrics['fn']:2d}, TN={metrics['tn']:2d}, "
              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    for tau, metrics in so["pronunciation_abstention_vs_word_misp"].items():
        print(f"  Abstain (misp)   tau={tau:<8}: TP={metrics['tp']:2d}, FP={metrics['fp']:2d}, FN={metrics['fn']:2d}, TN={metrics['tn']:2d}, "
              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    gr = json.loads((DATA / "rev_csharp_guardrails.json").read_text(encoding="utf-8"))
    sc = gr["strataCounts"]
    fc = gr["fallbackContractSuite"]
    print(f"\nC# Guardrail & Parser Contract Audit: parser tests={fc['passed']}/{fc['total']}, turns N={gr['totalTurns']}")
    for st in ("A", "B", "C", "D", "ALL"):
        s = sc[st]
        print(f"  Stratum {st:<3} (n={s['total']:3d}): raw_corr={s['raw_has_correction']:3d}, raw_quests={s['raw_has_quests']:3d}, "
              f"r1={s['r1_score_clamp_over_corr']}, r2={s['r2_fragment_score_capped']}, "
              f"r3={s['r3_objective_credit_cleared']}, r4={s['r4_unanchored_correction_dropped']}, nonASCII={s['non_ascii_reply_chars']}")

    print("\nALL OFFLINE ASSERTIONS & TABLE VERIFICATIONS PASSED.")


if __name__ == "__main__":
    main()
