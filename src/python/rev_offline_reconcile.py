"""Offline reconciliation, McNemar 2x2 tables, WER word error counts, canonicalization audit,
denominator reconciliation, pre-buffer / L_min baselines on archived Ryzen 5 5600 timings,
and cyclic vs disjoint long-reply breakdown.
"""
import collections
import csv
import json
import os
import re
import sys

import numpy as np
from num2words import num2words
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_common as E

RESULTS = E.RESULTS
GAP_MS = 20.0

UK_US = {
    "practise": "practice",
    "favourite": "favorite",
    "centre": "center",
    "colour": "color",
}
COMPOUND = [
    ("take out", "takeout"),
    ("key card", "keycard"),
    ("wi fi", "wifi"),
]


def norm_raw(text: str):
    """Minimal lowercase + apostrophe unify + strip non-alphanumeric punctuation."""
    t = (text or "").lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", t).split()


def norm_canon(text: str):
    """Canonical normalization used for E2 semantic/orthographic equivalence checks."""
    t = (text or "").lower().replace("’", "'").replace("-", " ")
    t = re.sub(r"\$(\d+)", r"\1 dollars", t)
    t = re.sub(r"\d+(\.\d+)?", lambda m: num2words(float(m.group()) if "." in m.group() else int(m.group())), t)
    t = re.sub(r"[^a-z' ]+", " ", t)
    for src, dst in COMPOUND:
        t = re.sub(r"\b" + re.escape(src) + r"\b", dst, t)
    return [UK_US.get(w, w) for w in t.split()]


def lev_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], op[i][j] = min(choices, key=lambda x: x[0])
    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == "M":
            i -= 1
            j -= 1
        elif cur == "S":
            s += 1
            i -= 1
            j -= 1
        elif cur == "D":
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins, dp[n][m]


def gaps_of(chunk_timings, pre_buffer_ms=0.0):
    t_synth = 0.0
    t_play = 0.0
    gaps = []
    ttfa = None
    for idx, (s_ms, a_ms) in enumerate(chunk_timings):
        t_synth += s_ms
        if idx == 0:
            t_play = t_synth + pre_buffer_ms
            ttfa = t_play
        else:
            if t_play < t_synth:
                gaps.append(t_synth - t_play)
                t_play = t_synth
            else:
                gaps.append(0.0)
        t_play += a_ms
    return ttfa, gaps, t_play


def desc_stats(arr):
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def main():
    out = {}

    # ── 1. E1 Denominators & Archived Baselines (Ryzen 5 5600) ───────────────
    e1_replies = list(csv.DictReader(open(os.path.join(RESULTS, "e1_replies.csv"), encoding="utf-8")))
    timing = {
        r["text"]: (float(r["synth_ms"]), float(r["audio_ms"]))
        for r in csv.DictReader(open(os.path.join(RESULTS, "e1_synth.csv"), encoding="utf-8"))
    }
    turns = E.load_turns()
    reply_texts = {f"{t['run']}|{t['turn']:03d}": E.reply_of(t["raw"]).strip() for t in turns}
    all_ids = sorted(reply_texts.keys())
    multi_ids = [rid for rid in all_ids if len(E.sentences(reply_texts[rid])) >= 2]
    single_ids = [rid for rid in all_ids if len(E.sentences(reply_texts[rid])) == 1]

    denom_check = {}
    for st in ("W", "A", "S"):
        all_rows = [r for r in e1_replies if r["strategy"] == st]
        m_rows = [r for r in all_rows if int(r["sentences"]) >= 2]
        aff_m = sum(1 for r in m_rows if int(r["gaps_over_20ms"]) > 0)
        tot_g = sum(int(r["gaps_over_20ms"]) for r in m_rows)
        denom_check[st] = {
            "total_replies": len(all_rows),
            "single_sentence": len(all_rows) - len(m_rows),
            "multi_sentence": len(m_rows),
            "affected_replies": aff_m,
            "gap_events_over_20ms": tot_g,
            "pct_of_multi_484": 100.0 * aff_m / len(m_rows),
            "pct_of_all_512": 100.0 * aff_m / len(all_rows),
            "ttfa_multi_ms": desc_stats([float(r["ttfa_ms"]) for r in m_rows]),
            "ttfa_all_ms": desc_stats([float(r["ttfa_ms"]) for r in all_rows]),
            "completion_multi_ms": desc_stats([float(r["completion_ms"]) for r in m_rows]),
        }
    out["e1_denominator_reconciliation"] = denom_check

    # Pre-buffer baseline B on per-sentence chunks (100% matched across 484 Ryzen replies)
    prebuf = {}
    for b_ms in (0, 100, 200, 300, 400, 500, 600, 800, 1000):
        ttfas, aff, gtot, comps = [], 0, 0, []
        for rid in multi_ids:
            chs = E.sentences(reply_texts[rid])
            ttfa, gs, comp = gaps_of([timing[c] for c in chs], pre_buffer_ms=b_ms)
            over = sum(1 for g in gs if g > GAP_MS)
            ttfas.append(ttfa)
            comps.append(comp)
            if over > 0:
                aff += 1
                gtot += over
        prebuf[f"S_buf_{b_ms}ms"] = {
            "pre_buffer_ms": b_ms,
            "n_multi": len(multi_ids),
            "affected_replies": aff,
            "pct_multi_484": 100.0 * aff / len(multi_ids),
            "pct_all_512": 100.0 * aff / len(all_ids),
            "gap_events": gtot,
            "ttfa_ms": desc_stats(ttfas),
            "completion_ms": desc_stats(comps),
        }
    out["e1_ryzen_prebuffer_baselines"] = prebuf

    # L_min tuned prefix baselines on matched Ryzen cache subset
    def lmin_prefix(text, lmin):
        sents = E.sentences(text)
        chs = [sents[0]]
        for s in sents[1:]:
            if len(chs[0]) < lmin and len(chs) == 1:
                chs[0] = chs[0] + " " + s
            else:
                chs.append(s)
        return chs

    lmin_res = {}
    for lmin in (40, 60, 80, 100, 120):
        ttfas, aff, gtot, comps, matched_n, uncached_rids = [], 0, 0, [], 0, []
        for rid in multi_ids:
            chs = lmin_prefix(reply_texts[rid], lmin)
            if all(c in timing for c in chs):
                matched_n += 1
                ttfa, gs, comp = gaps_of([timing[c] for c in chs], 0.0)
                over = sum(1 for g in gs if g > GAP_MS)
                ttfas.append(ttfa)
                comps.append(comp)
                if over > 0:
                    aff += 1
                    gtot += over
            else:
                uncached_rids.append(rid)
        lmin_res[f"L_min_{lmin}"] = {
            "lmin_chars": lmin,
            "matched_in_ryzen_cache": matched_n,
            "uncached_count": len(uncached_rids),
            "uncached_rids": uncached_rids,
            "affected_replies_cached": aff,
            "pct_cached": 100.0 * aff / matched_n if matched_n else None,
            "ttfa_ms": desc_stats(ttfas),
            "completion_ms": desc_stats(comps),
        }
    out["e1_ryzen_lmin_cached_subset"] = lmin_res

    # Long-reply cyclic 64 vs disjoint 32 (archived cache no-cross-seam)
    three_plus = [rid for rid in multi_ids if len(E.sentences(reply_texts[rid])) >= 3]
    cyc_pairs = [(three_plus[i], three_plus[(i + 1) % len(three_plus)]) for i in range(len(three_plus))]
    dis_pairs = [(three_plus[i], three_plus[i + 1]) for i in range(0, len(three_plus), 2)]

    def eval_pair_set(pairs, label):
        lw, la, ls, lga, lgs, char_lens, sent_lens = [], [], [], [], [], [], []
        for r1_id, r2_id in pairs:
            r1, r2 = reply_texts[r1_id], reply_texts[r2_id]
            sent_lens.append(len(E.sentences(r1)) + len(E.sentences(r2)))
            char_lens.append(len(r1) + 1 + len(r2))
            ch_a = E.adaptive_chunks(r1) + E.adaptive_chunks(r2)
            ch_s = E.sentences(r1) + E.sentences(r2)
            lw.append(timing[r1][0] + timing[r2][0])
            t_a, g_a, _ = gaps_of([timing[c] for c in ch_a])
            t_s, g_s, _ = gaps_of([timing[c] for c in ch_s])
            la.append(t_a)
            ls.append(t_s)
            lga.append(sum(1 for g in g_a if g > GAP_MS))
            lgs.append(sum(1 for g in g_s if g > GAP_MS))
        return {
            "label": label,
            "n_pairs": len(pairs),
            "source_replies": len(three_plus),
            "sentence_count_dist": dict(sorted(collections.Counter(sent_lens).items())),
            "chars": desc_stats(char_lens),
            "W_ttfa_ms": desc_stats(lw),
            "A_ttfa_ms": desc_stats(la),
            "S_ttfa_ms": desc_stats(ls),
            "A_affected_replies": sum(1 for x in lga if x > 0),
            "A_affected_pct": 100.0 * sum(1 for x in lga if x > 0) / len(pairs),
            "S_affected_replies": sum(1 for x in lgs if x > 0),
            "S_affected_pct": 100.0 * sum(1 for x in lgs if x > 0) / len(pairs),
        }

    out["e1_long_reply_breakdown"] = {
        "cyclic_64_no_cross_seam": eval_pair_set(cyc_pairs, "64 cyclic pairs (each source reply appears once as prefix and once as suffix, 0 self-pairs)"),
        "disjoint_32_no_cross_seam": eval_pair_set(dis_pairs, "32 disjoint non-overlapping pairs"),
    }

    # ── 2. E2 Detailed McNemar 2x2 Tables, Word Errors, and Error Inventory ─────
    e2_turns = list(csv.DictReader(open(os.path.join(RESULTS, "e2_turns.csv"), encoding="utf-8")))
    by_c = {"C1": {}, "C2": {}, "C3": {}}
    for r in e2_turns:
        by_c[r["condition"]][r["uid"]] = r
    uids = sorted(by_c["C1"].keys())

    e2_detail = {}
    for norm_name, norm_fn in [("raw", norm_raw), ("canonical", norm_canon)]:
        cond_stats = {}
        exact_map = {"C1": {}, "C2": {}, "C3": {}}
        for c in ("C1", "C2", "C3"):
            tot_s = tot_d = tot_i = tot_ed = tot_ref = exact_n = 0
            for u in uids:
                ref = norm_fn(by_c[c][u]["utterance"])
                hyp = norm_fn(by_c[c][u]["transcript"])
                s, d, ins, ed = lev_counts(ref, hyp)
                tot_s += s
                tot_d += d
                tot_i += ins
                tot_ed += ed
                tot_ref += len(ref)
                ok = ref == hyp
                exact_map[c][u] = ok
                if ok:
                    exact_n += 1
            cond_stats[c] = {
                "exact_matches": exact_n,
                "total": len(uids),
                "exact_pct": 100.0 * exact_n / len(uids),
                "ref_words": tot_ref,
                "substitutions": tot_s,
                "deletions": tot_d,
                "insertions": tot_i,
                "total_word_errors": tot_ed,
                "wer_pct": 100.0 * tot_ed / tot_ref,
            }
        mcnemar = {}
        for ca, cb in (("C1", "C2"), ("C1", "C3"), ("C2", "C3")):
            n11 = sum(1 for u in uids if exact_map[ca][u] and exact_map[cb][u])
            n10 = sum(1 for u in uids if exact_map[ca][u] and not exact_map[cb][u])
            n01 = sum(1 for u in uids if not exact_map[ca][u] and exact_map[cb][u])
            n00 = sum(1 for u in uids if not exact_map[ca][u] and not exact_map[cb][u])
            disc = n10 + n01
            p_exact = float(stats.binomtest(min(n10, n01), disc, 0.5).pvalue) if disc > 0 else 1.0
            mcnemar[f"{ca}_vs_{cb}"] = {
                "n11_both_exact": n11,
                "n10_only_first_exact": n10,
                "n01_only_second_exact": n01,
                "n00_neither_exact": n00,
                "n10_uids": [u for u in uids if exact_map[ca][u] and not exact_map[cb][u]],
                "n01_uids": [u for u in uids if not exact_map[ca][u] and exact_map[cb][u]],
                "n00_uids": [u for u in uids if not exact_map[ca][u] and not exact_map[cb][u]],
                "exact_binomial_p": p_exact,
            }
        e2_detail[norm_name] = {"conditions": cond_stats, "mcnemar_2x2": mcnemar}

    # Complete utterance-by-condition diff inventory
    inventory = []
    for u in uids:
        utt = by_c["C1"][u]["utterance"]
        row_diffs = {}
        any_raw_diff = False
        for c in ("C1", "C2", "C3"):
            hyp = by_c[c][u]["transcript"]
            raw_eq = norm_raw(utt) == norm_raw(hyp)
            can_eq = norm_canon(utt) == norm_canon(hyp)
            if not raw_eq:
                any_raw_diff = True
            row_diffs[c] = {
                "transcript": hyp,
                "raw_exact": raw_eq,
                "canon_exact": can_eq,
                "score": by_c[c][u]["score"],
            }
        if any_raw_diff:
            inventory.append({"uid": u, "scenario": by_c["C1"][u]["scenario"], "utterance": utt, "conditions": row_diffs})
    e2_detail["utterance_diff_inventory"] = inventory
    out["e2_transcription_reconciliation"] = e2_detail

    out_path = os.path.join(RESULTS, "rev_offline_reconciled.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print("E1 denominators:", json.dumps(denom_check, indent=2))
    print("E1 prebuffer Pareto:", {k: (v['affected_replies'], round(v['pct_multi_484'], 2), round(v['ttfa_ms']['median'], 1)) for k, v in prebuf.items()})
    print("E1 L_min coverage & gaps:", {k: (v['matched_in_ryzen_cache'], v['affected_replies_cached'], round(v['ttfa_ms']['median'], 1)) for k, v in lmin_res.items()})
    print("E2 Canon McNemar C1 vs C2:", e2_detail["canonical"]["mcnemar_2x2"]["C1_vs_C2"])
    print("E2 Diff inventory count:", len(inventory))


if __name__ == "__main__":
    main()
