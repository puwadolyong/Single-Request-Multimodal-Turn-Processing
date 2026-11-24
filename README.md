# Reproducibility Archive — Single-Request Multimodal Turn Processing & Adaptive On-Device TTS



## 1. Exact Pinned Model, Voice, Runtime, and Hardware Identifiers

| Component | Identifier / Configuration |
| :--- | :--- |
| Experiment 1 logged corpus model (10 Sep 2026) | `gemini-3.5-flash` (typed input; history reset at 8 [`run1`] and 32 [`ceiling1`] turns) |
| Experiment 2 primary production model (14 Oct 2026 & tb0 ablations) | `gemini-3.6-flash` (`temperature=0.9`, `propertyOrdering` declared, `thinkingConfig.thinkingBudget=0`, `maxOutputTokens=2048` in zero-truncation ablations vs legacy `800`) |
| Separate Gemini STT prompt (`gemini-3.6-flash` / `gemini-3.5-transcribe`) | `temperature=0.0`, `maxOutputTokens=200`, prompt in `prompts/transcribe_prompt.txt` |
| Cloud TTS baseline (`C3`) | `gemini-3.8-flash-tts` (prebuilt voice `Kore`) |
| Cross-model portability comparison (`n=32`) | `gemini-3.6-flash`, `gemini-3.8-flash`, `gemini-3.5-flash-lite` (`thinkingBudget=0`) |
| Local dedicated ASR baseline (`faster-whisper`) | `base.en` and `small.en` (`device="cpu"`, `compute_type="int8"`, `cpu_threads=4`, `beam_size=1`) |
| On-device neural TTS voice & runtime | `Kokoro-82M (v0.19)` (`model.onnx`, `voices.bin`, `tokens.txt`), `sherpa-onnx 1.13.8`, speaker `sid=0`, rate `speed=1.05`, `4` CPU threads (`2` threads pinned to cores `0–3` on constrained laptop profile) |
| Client engine & harness | Unity `6000.5.3f1` / `.NET 8.0` WinMM `waveOut` C# harness (`src/dotnet_harness/Program.cs`), 16 kHz mono PCM input, 24 kHz synth resampled to 48 kHz playback |
| Reference workstation (E1/E2 Table 4) & revision profiling host | Reference workstation (`Table 4`, `E1`): AMD Ryzen 5 5600 (6 cores, Windows); Revision instrumentation, `live-e2e` (`n=24`), `N=128` WinMM profiles, and `tb0` (`Table 3` / `SpeechOcean762`): Intel Core Ultra 7 265K, 31.37 GB RAM, RTX 5070 Ti, Windows |
| Manuscript-to-key naming map | Profiles (`rev_native_playback_128.csv`, `rev_chunking_baselines.json`): `Desktop 4t (idle)` -> `desktop_4t_idle`, `Desktop 4t (8-thread render load)` -> `desktop_4t_render_load`, `Constrained 2t (affinity 0-3)` -> `laptop_2t_affinity4`; Policies: `A_causal` -> `CausalOnline` |
| Random seeds | Bootstrap CIs: `seed=42` (`5,000` cluster/utterance resamples); `SpeechOcean762` stratified sample (`N=100`): `random.Random(42)` |

---

## 2. Offline Reproducible vs. Hosted-Service Experiments

Following Reviewer 1's request, every result in the manuscript is explicitly classified below by whether it reproduces **100% deterministically offline** or requires **live calls to a hosted API**:

### A. 100% Offline Reproducible (No API Key, No Network)
1. **All Manuscript Tables & Statistical Tests from Archived Logs**: Run `python reproduce_tables.py` inside this folder. It recomputes:
   - Experiment 1 denominator reconciliation (`n=484` multi-sentence / `N=512` total), Strategy `W/S/A`, `alpha` sensitivity, `S+B` prebuffers, `L_min` cached subset, cyclic-64 vs disjoint-32 long-pair breakdowns from `data/e1_replies.csv`, `data/e1_synth.csv`, `data/e3_alpha.csv`, and `data/rev_offline_reconciled.json`.
   - C# `SpeechChunker` / `ParsedResponse` guardrail audit (`512` turns by Stratum A/B/C/D) and fallback parser contract test suite (`15/15` passed via `SalvageJsonReplyField`) from `data/rev_csharp_guardrails.json`.
   - Instrumented C# WinMM client timing breakdown (`n=64` warm + `5` cold starts) from `data/rev_instrument_e2e.json`, plus continuous single-stopwatch push-to-talk-to-`waveOutWrite` live turn validation (`n=24`) from `data/rev_live_client_e2e_24.json` comparing wall-clock `waveOutWrite` onset directly against Table 3's additive reconstruction formula (`api + synth0 + 9.0 + 8.33 ms`).
   - `1x` wall-clock `60 Hz` `waveOut` critical validation (`n=24`) from `data/rev_realtime_60hz_validation.json`.
   - Stratified `N=128` native C# playback (`Desktop 4t (idle)` = `desktop_4t_idle`, `Desktop 4t (8-thread render load)` = `desktop_4t_render_load`, `Constrained 2t (affinity 0-3)` = `laptop_2t_affinity4`), reconciled zero-fallback causal/baseline chunking sweep (`S+B`, `L_min`, `Lookahead1`, `A_causal` = `CausalOnline`, `A`, `A_post`, `32` disjoint pairs), exact Clopper–Pearson 95% CIs and McNemar `p`-values, and speed/CPU slowdown sensitivity (`1.30x` voice speed with unadjusted `delta=52` and speed-scaled `delta(1.30)=42`, plus `1.35x` CPU slowdown) from `data/rev_native_playback_128.csv` and `data/rev_chunking_baselines.json`.
   - Experiment 2 raw and canonical `S/D/I` WER counts, exact `2x2` McNemar contingency tables (`C1` vs `C2` vs `C3`), canonicalization rules, and `19`-utterance error inventory from `data/e2_turns.csv` and `data/rev_offline_reconciled.json`.
   - Revision live-API ablation statistics (paired `n=64` zero-thinking `C1/C1_rf/C2_flash/C2_flash_stream/C4`, counterfactual `C4_base`, paired Wilcoxon tests/bootstrap CIs in `part2_wilcoxon_paired_64` and `part3b_wilcoxon_paired_24`, default-thinking `n=51` `3.5-transcribe` subset, history depth `h=0/8/20`, network profiles LAN/Wi-Fi/LTE, model portability `n=32`, and real human `SpeechOcean762` `N=100` overall/per-stratum raw & canonical WER, word- and utterance-prevalence-weighted WER, correlation, and abstention metrics) from `data/rev_live_api_ablations*.json` and `data/rev_speechocean762.json`.
2. **Re-running Local On-Device TTS & Whisper Inference**:
   - Running `src/python/rev_chunking_un_cached.py` and the `faster-whisper` stage of `src/python/rev_e2_asr_streaming_ablation.py` requires only local Kokoro-82M ONNX files and CPU execution (wall-clock synthesis durations vary smoothly with host CPU speed).
   - Running `src/dotnet_harness/Program.cs` (`guardrails`, `instrument`, `playback`, `realtime`) requires `.NET 8.0` + `sherpa-onnx` native binaries and local Kokoro-82M files.

### B. Hosted Cloud API Dependent (Subject to Provider Backend Changes)
- Calling `e2_turn_latency.py`, `rev_e2_asr_streaming_ablation.py` (Gemini conditions), `rev_e5_speechocean.py` (Gemini conditions), or `dotnet run ... -- live-e2e` sends live HTTPS requests to Google's `generativelanguage.googleapis.com` API using an environment variable `GEMINI_API_KEY`.
- Because hosted commercial models undergo server-side version updates, capacity queuing, and rate limits (e.g., `gemini-3.5-transcribe` Tier-1 `100 RPD` cap), live re-execution reproduces qualitative architectural trends while exact millisecond distributions reflect current provider backend conditions. Full raw HTTP transcripts and timing checkpoints (`data/e2_calls.csv`, `data/rev_e2_tb0_checkpoint.jsonl`, `data/rev_e5_speechocean_ckpt.jsonl`) are preserved in this archive so all downstream analyses run offline without re-querying the API.

---

## 3. Quick Start (Offline Verification)

```bash
git clone https://github.com/puwadolyong/Single-Request-Multimodal-Turn-Processing.git
cd Single-Request-Multimodal-Turn-Processing
python reproduce_tables.py
```

Requires Python 3.10+ (`numpy`, `scipy`, `jiwer`, `num2words` only needed if re-scoring raw strings from scratch; `reproduce_tables.py` itself uses only the Python standard library to load and verify all archived CSVs/JSONs).

---

## 4. Directory Layout

```
Single-Request-Multimodal-Turn-Processing/
├── LICENSE
├── README.md
├── reproduce_tables.py          # Offline verification script for all manuscript tables
├── prompts/
│   ├── coach_prompt_body.txt
│   ├── transcribe_prompt.txt
│   ├── scenarios.json
│   └── example_system_prompt_FreeTalk.txt
├── schemas/
│   ├── CoachTurnSchema.json
│   └── property_ordering.json
├── src/
│   ├── csharp/                  # Production Unity client & proxy source files
│   ├── dotnet_harness/          # Native .NET 8 / sherpa-onnx / WinMM waveOut benchmark
│   └── python/                  # Sanitized experiment & analysis scripts
└── data/
    ├── audio/                   # 64 synthetic 16 kHz learner WAVs (u00..u63)
    ├── e1_*.csv, e2_*.csv, e3_alpha.csv
    ├── turns_run1.jsonl, turns_run2.jsonl
    └── rev_*.json, rev_*.csv, rev_*.jsonl
```
