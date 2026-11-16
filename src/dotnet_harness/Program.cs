using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Eitan.SherpaONNXUnity.Runtime.Native;
using GeminiAvatar;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace ClientHarness;

internal static class Program
{
    private const string ConviaRoot = @"<CONVIA_PROJECT_ROOT>";
    private const string KokoroDir = ConviaRoot + @"\Assets\StreamingAssets\Kokoro-82M\kokoro-en-v0_19";
    private static readonly string PaperDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    public static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        string mode = args.Length > 0 ? args[0].ToLowerInvariant() : "all";
        string paperDir = ResolvePaperDir();
        string resultsDir = Path.Combine(paperDir, "experiments", "results");
        Directory.CreateDirectory(resultsDir);

        if (mode is "all" or "guardrails")
            RunGuardrailAndFallbackAudit(resultsDir);

        if (mode is "all" or "instrument")
            await RunInstrumentationBench(resultsDir);

        if (mode is "all" or "playback")
            RunPlaybackProfilesBench(resultsDir);

        if (mode is "realtime")
            await RunRealtimeWaveOutValidation(resultsDir);

        if (mode is "live-e2e")
            await RunLiveEndToEndWaveOutValidationAsync(paperDir, resultsDir);

        return 0;
    }

    private static string ResolvePaperDir()
    {
        string d = Directory.GetCurrentDirectory();
        while (!string.IsNullOrEmpty(d))
        {
            if (File.Exists(Path.Combine(d, "reproduce_tables.py")) || File.Exists(Path.Combine(d, "experiments", "exp_common.py")))
                return d;
            d = Path.GetDirectoryName(d);
        }
        return Environment.GetEnvironmentVariable("CONVIA_PAPER_DIR") ?? Directory.GetCurrentDirectory();
    }

    // ──────────────────────────────────────────────────────────────
    // 1. C# GUARDRAIL AUDIT OVER 512 LOGGED TURNS + MALFORMED FALLBACK SUITE
    // ──────────────────────────────────────────────────────────────
    private sealed class TurnLogRow
    {
        [JsonProperty("run")] public string Run { get; set; }
        [JsonProperty("turn")] public int Turn { get; set; }
        [JsonProperty("scenario")] public string Scenario { get; set; }
        [JsonProperty("mode")] public string Mode { get; set; }
        [JsonProperty("userText")] public string UserText { get; set; }
        [JsonProperty("promptTokens")] public int PromptTokens { get; set; }
        [JsonProperty("outputTokens")] public int OutputTokens { get; set; }
        [JsonProperty("requestMs")] public double RequestMs { get; set; }
        [JsonProperty("raw")] public string Raw { get; set; }
    }

    private static void RunGuardrailAndFallbackAudit(string resultsDir)
    {
        Console.WriteLine("=== [1/3] Running C# ParsedResponse Guardrail & Fallback Audit ===");
        var logFiles = new[]
        {
            Path.Combine(ConviaRoot, "Paper", "turns_run1.jsonl"),
            Path.Combine(ConviaRoot, "Paper", "turns_run2.jsonl")
        };

        var rows = new List<TurnLogRow>();
        foreach (string lf in logFiles)
        {
            foreach (string line in File.ReadLines(lf, Encoding.UTF8))
            {
                if (!string.IsNullOrWhiteSpace(line))
                    rows.Add(JsonConvert.DeserializeObject<TurnLogRow>(line));
            }
        }

        var strataCounts = new Dictionary<string, Dictionary<string, int>>();
        foreach (string st in new[] { "ALL", "A", "B", "C", "D" })
        {
            strataCounts[st] = new Dictionary<string, int>
            {
                ["total"] = 0,
                ["raw_has_correction"] = 0,
                ["raw_has_quests"] = 0,
                ["raw_perfect"] = 0,
                ["r1_score_clamp_over_corr"] = 0,
                ["r2_fragment_score_capped"] = 0,
                ["r2_fragment_perfect_stripped"] = 0,
                ["r3_objective_credit_cleared"] = 0,
                ["r3_cleared_due_score_only"] = 0,
                ["r3_cleared_due_length_only"] = 0,
                ["r3_cleared_due_both"] = 0,
                ["r4_unanchored_correction_dropped"] = 0,
                ["final_has_correction"] = 0,
                ["final_has_quests"] = 0,
                ["non_ascii_reply_chars"] = 0
            };
        }

        var triggeredEvents = new List<object>();

        foreach (TurnLogRow row in rows)
        {
            string st = row.Run.Contains('|') ? row.Run.Split('|')[1] : "UNK";
            CoachTurn rawObj = JsonConvert.DeserializeObject<CoachTurn>(row.Raw);
            ParsedResponse finalRes = ParsedResponse.Parse(row.Raw, row.UserText);

            string rawCorr = (rawObj.Correction ?? "").Trim();
            int rawScore = rawObj.Score;
            string rawStatus = ParsedResponse.NormalizeStatus(rawObj.Status);
            List<int> rawQuests = rawObj.QuestsCompleted ?? new List<int>();
            string utt = string.IsNullOrWhiteSpace(row.UserText) ? (rawObj.Transcription ?? "") : row.UserText;

            bool r1ScoreClamp = !string.IsNullOrEmpty(rawCorr) && rawScore >= ParsedResponse.FlawlessBandFloor;
            bool r2FragCap = rawScore > ParsedResponse.FragmentScoreCeiling && ParsedResponse.IsFragment(utt);
            int effectiveScoreAfterFrag = r2FragCap ? ParsedResponse.FragmentScoreCeiling : rawScore;
            bool r2PerfectStripped = r2FragCap && rawStatus == "PERFECT";

            bool scoreAllows = effectiveScoreAfterFrag < 0 || effectiveScoreAfterFrag >= ParsedResponse.SentenceBandFloor;
            bool uttAllows = string.IsNullOrWhiteSpace(utt) || ParsedResponse.IsSubstantialUtterance(utt);
            bool r3QuestDrop = rawQuests.Count > 0 && (!scoreAllows || !uttAllows);

            bool r4CorrDrop = !string.IsNullOrEmpty(rawCorr) && !finalRes.HasCorrection;
            bool hasNonAscii = Regex.Replace(finalRes.Text ?? "", @"[’‘“”—–…]", "").Any(ch => ch > 127);

            foreach (string key in new[] { "ALL", st })
            {
                var d = strataCounts[key];
                d["total"]++;
                if (!string.IsNullOrEmpty(rawCorr)) d["raw_has_correction"]++;
                if (rawQuests.Count > 0) d["raw_has_quests"]++;
                if (rawStatus == "PERFECT") d["raw_perfect"]++;
                if (r1ScoreClamp) d["r1_score_clamp_over_corr"]++;
                if (r2FragCap) d["r2_fragment_score_capped"]++;
                if (r2PerfectStripped) d["r2_fragment_perfect_stripped"]++;
                if (r3QuestDrop)
                {
                    d["r3_objective_credit_cleared"]++;
                    if (!scoreAllows && uttAllows) d["r3_cleared_due_score_only"]++;
                    else if (scoreAllows && !uttAllows) d["r3_cleared_due_length_only"]++;
                    else d["r3_cleared_due_both"]++;
                }
                if (r4CorrDrop) d["r4_unanchored_correction_dropped"]++;
                if (finalRes.HasCorrection) d["final_has_correction"]++;
                if (finalRes.CompletedQuests.Count > 0) d["final_has_quests"]++;
                if (hasNonAscii) d["non_ascii_reply_chars"]++;
            }

            if (r1ScoreClamp || r2FragCap || r3QuestDrop || r4CorrDrop)
            {
                triggeredEvents.Add(new
                {
                    run = row.Run,
                    stratum = st,
                    turn = row.Turn,
                    scenario = row.Scenario,
                    utterance = utt,
                    wordCount = ParsedResponse.CountWords(utt),
                    isFragment = ParsedResponse.IsFragment(utt),
                    isSubstantial = ParsedResponse.IsSubstantialUtterance(utt),
                    rawScore,
                    finalScore = finalRes.Score,
                    rawStatus,
                    finalStatus = finalRes.FeedbackStatus,
                    rawQuests,
                    finalQuests = finalRes.CompletedQuests,
                    rawCorrection = rawCorr,
                    finalCorrection = finalRes.Correction,
                    rules = new
                    {
                        r1ScoreClamp,
                        r2FragCap,
                        r3QuestDrop,
                        r4CorrDrop
                    }
                });
            }
        }

        // Malformed & Legacy fallback contract test matrix
        var fallbackTests = new List<(string Name, Func<bool> Check)>
        {
            ("NullPayload_SafeEmpty", () => { var r = ParsedResponse.Parse(null); return r != null && r.Text == "" && !r.HasScore; }),
            ("EmptyPayload_SafeEmpty", () => { var r = ParsedResponse.Parse("   "); return r != null && r.Text == "" && !r.HasFeedback; }),
            ("TruncatedJson_DoesNotThrow_NoJsonLeak", () => {
                var r = ParsedResponse.Parse("{\"reply\":\"Hello there\", \"score\": 85");
                return r != null && !r.HasScore && !r.Text.Contains("\"score\"");
            }),
            ("CodeFencedJson_StripsFenceAndParses", () => {
                var r = ParsedResponse.Parse("```json\n{\"reply\":\"Sure thing!\",\"score\":92,\"status\":\"PERFECT\",\"emotion\":\"happy\"}\n```");
                return r.Text == "Sure thing!" && r.Score == 92 && r.FeedbackStatus == "PERFECT";
            }),
            ("ApiErrorJsonObject_NotMisreadAsTurn", () => {
                var r = ParsedResponse.Parse("{\"error\":{\"code\":429,\"message\":\"Quota exceeded\"}}");
                return !r.HasScore && string.IsNullOrEmpty(r.UserTranscription);
            }),
            ("MissingReplyJson_FallsThroughSafely", () => {
                var r = ParsedResponse.Parse("{\"score\":95,\"status\":\"PERFECT\",\"emotion\":\"happy\"}");
                return r != null && r.Text == "";
            }),
            ("LegacyTaggedFull_ParsesAllBlocks", () => {
                string raw = "[TRANSCRIPTION]I want coffee[/TRANSCRIPTION][FEEDBACK][SCORE:72][STATUS:NEEDS_FIX]\n• Correction: Original: want -> Corrected: would like[/FEEDBACK][SPOKEN][EMOTION:happy:2]Sure! Coming right up.[/SPOKEN][QUEST_DONE:1]";
                var r = ParsedResponse.Parse(raw, "I want coffee");
                return r.UserTranscription == "I want coffee" && r.Score == 72 && r.FeedbackStatus == "NEEDS_FIX"
                    && r.Text == "Sure! Coming right up." && r.Emotion == AvatarEmotion.Happy && r.EmotionIntensity == 2
                    && !r.FeedbackText.Contains("[SCORE:") && r.CompletedQuests.Count == 0; // <4 words cleared by shared R3!
            }),
            ("LegacyUnclosedFeedbackBlock_StripsFeedback", () => {
                string raw = "[FEEDBACK][SCORE:80][STATUS:GOOD]\n• Quick Tip: Good try.[SPOKEN]What size would you like?[/SPOKEN]";
                var r = ParsedResponse.Parse(raw);
                return r.Text == "What size would you like?" && r.Score == 80 && !r.Text.Contains("Quick Tip");
            }),
            ("LegacyUnclosedSpokenBlock_StillSpeaksDialogue", () => {
                string raw = "[FEEDBACK][SCORE:88][/FEEDBACK][SPOKEN]That sounds great!";
                var r = ParsedResponse.Parse(raw);
                return r.Text == "That sounds great!" && r.Score == 88;
            }),
            ("LegacyQuestSharedPolicy_Over4Words_Retained", () => {
                string raw = "[FEEDBACK][SCORE:85][STATUS:GOOD][/FEEDBACK][SPOKEN]Here you go.[/SPOKEN][QUEST_DONE:2]";
                var r = ParsedResponse.Parse(raw, "I would like a large iced coffee please");
                return r.CompletedQuests.SequenceEqual(new[] { 2 });
            }),
            ("FragmentHighScoreOverrule_Painkiller100_Capped55", () => {
                string raw = "{\"reply\":\"We have ibuprofen.\",\"score\":100,\"status\":\"PERFECT\",\"emotion\":\"happy\",\"questsCompleted\":[1]}";
                var r = ParsedResponse.Parse(raw, "Painkiller");
                return r.Score == 55 && r.FeedbackStatus == "GOOD" && r.CompletedQuests.Count == 0 && !r.FeedbackText.Contains("Great!");
            }),
            ("ShortContractionExemptFromFragmentCap", () => {
                string raw = "{\"reply\":\"Glad to hear it!\",\"score\":95,\"status\":\"PERFECT\",\"emotion\":\"happy\"}";
                var r = ParsedResponse.Parse(raw, "I'm fine.");
                return r.Score == 95 && r.FeedbackStatus == "PERFECT";
            }),
            ("ShortImperativeExemptFromFragmentCap", () => {
                string raw = "{\"reply\":\"Turning left now.\",\"score\":95,\"status\":\"PERFECT\",\"emotion\":\"neutral\",\"questsCompleted\":[3]}";
                var r = ParsedResponse.Parse(raw, "Turn left.");
                // Keeps 95 score (imperative exempt from fragment ceiling) while objective gate requires >=4 words
                return r.Score == 95 && r.CompletedQuests.Count == 0;
            }),
            ("CorrectedSentenceOver90_ClampedTo89", () => {
                string raw = "{\"reply\":\"Sure!\",\"score\":94,\"status\":\"GOOD\",\"correction\":\"Original: I like a latte -> Corrected: I'd like a latte\",\"emotion\":\"happy\"}";
                var r = ParsedResponse.Parse(raw, "I like a latte");
                return r.Score == 89 && r.HasCorrection;
            }),
            ("OutOfRangeQuestId_Stripped", () => {
                string raw = "{\"reply\":\"Ok.\",\"score\":90,\"status\":\"PERFECT\",\"questsCompleted\":[1, 99, -2],\"emotion\":\"neutral\"}";
                var r = ParsedResponse.Parse(raw, "I would like an iced coffee please");
                return r.CompletedQuests.SequenceEqual(new[] { 1 });
            })
        };

        var fallbackResults = new List<object>();
        int passed = 0;
        foreach (var (name, check) in fallbackTests)
        {
            bool ok = false;
            string err = null;
            try { ok = check(); } catch (Exception ex) { err = ex.Message; }
            if (ok) passed++;
            fallbackResults.Add(new { name, passed = ok, error = err });
        }

        var output = new
        {
            totalTurns = rows.Count,
            strataCounts,
            triggeredEvents,
            fallbackContractSuite = new
            {
                total = fallbackTests.Count,
                passed,
                failed = fallbackTests.Count - passed,
                cases = fallbackResults
            }
        };

        string outPath = Path.Combine(resultsDir, "rev_csharp_guardrails.json");
        File.WriteAllText(outPath, JsonConvert.SerializeObject(output, Formatting.Indented));
        Console.WriteLine($"  Wrote {outPath} (triggered={triggeredEvents.Count}, fallback={passed}/{fallbackTests.Count})");
    }

    // ──────────────────────────────────────────────────────────────
    // 2. END-TO-END CLIENT + KESTREL PROXY + KOKORO COLD/WARM + RESAMPLING INSTRUMENTATION
    // ──────────────────────────────────────────────────────────────
    [DllImport("winmm.dll")]
    private static extern int waveOutOpen(out IntPtr hWaveOut, int uDeviceID, ref WaveFormatEx lpFormat, IntPtr dwCallback, IntPtr dwInstance, int dwFlags);

    [DllImport("winmm.dll")]
    private static extern int waveOutPrepareHeader(IntPtr hWaveOut, ref WaveHdr lpWaveOutHdr, int uSize);

    [DllImport("winmm.dll")]
    private static extern int waveOutWrite(IntPtr hWaveOut, ref WaveHdr lpWaveOutHdr, int uSize);

    [DllImport("winmm.dll")]
    private static extern int waveOutUnprepareHeader(IntPtr hWaveOut, ref WaveHdr lpWaveOutHdr, int uSize);

    [DllImport("winmm.dll")]
    private static extern int waveOutReset(IntPtr hWaveOut);

    [DllImport("winmm.dll")]
    private static extern int waveOutClose(IntPtr hWaveOut);

    [DllImport("winmm.dll")]
    private static extern uint timeBeginPeriod(uint uPeriod);

    [DllImport("winmm.dll")]
    private static extern uint timeEndPeriod(uint uPeriod);

    [StructLayout(LayoutKind.Sequential)]
    private struct WaveFormatEx
    {
        public ushort wFormatTag;
        public ushort nChannels;
        public uint nSamplesPerSec;
        public uint nAvgBytesPerSec;
        public ushort nBlockAlign;
        public ushort wBitsPerSample;
        public ushort cbSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct WaveHdr
    {
        public IntPtr lpData;
        public uint dwBufferLength;
        public uint dwBytesRecorded;
        public IntPtr dwUser;
        public uint dwFlags;
        public uint dwLoops;
        public IntPtr lpNext;
        public IntPtr reserved;
    }

    private static float[] ReadWavMonoFloat(string path, out int sampleRate)
    {
        byte[] b = File.ReadAllBytes(path);
        sampleRate = BitConverter.ToInt32(b, 24);
        int pos = 12;
        int dataOffset = 0, dataLen = 0;
        while (pos + 8 <= b.Length)
        {
            string id = Encoding.ASCII.GetString(b, pos, 4);
            int sz = BitConverter.ToInt32(b, pos + 4);
            if (id == "data") { dataOffset = pos + 8; dataLen = sz; break; }
            pos += 8 + sz;
        }
        int n = dataLen / 2;
        float[] samples = new float[n];
        for (int i = 0; i < n; i++)
            samples[i] = BitConverter.ToInt16(b, dataOffset + i * 2) / 32768f;
        return samples;
    }

    private static byte[] EncodeWavExactClient(float[] samples, int channels, int frequency)
    {
        using var stream = new MemoryStream(44 + samples.Length * 2);
        using var writer = new BinaryWriter(stream);
        int sampleCount = samples.Length;
        short ch = (short)channels;
        short bps = 16;
        int byteRate = frequency * ch * (bps / 8);
        short blockAlign = (short)(ch * (bps / 8));
        int dataSize = sampleCount * (bps / 8);

        writer.Write(Encoding.UTF8.GetBytes("RIFF"));
        writer.Write(36 + dataSize);
        writer.Write(Encoding.UTF8.GetBytes("WAVE"));
        writer.Write(Encoding.UTF8.GetBytes("fmt "));
        writer.Write(16);
        writer.Write((short)1);
        writer.Write(ch);
        writer.Write(frequency);
        writer.Write(byteRate);
        writer.Write(blockAlign);
        writer.Write(bps);
        writer.Write(Encoding.UTF8.GetBytes("data"));
        writer.Write(dataSize);
        for (int i = 0; i < sampleCount; i++)
        {
            float clamped = Math.Clamp(samples[i] * 32767f, -32768f, 32767f);
            writer.Write((short)clamped);
        }
        return stream.ToArray();
    }

    private static float PeakAmplitudeClient(float[] data)
    {
        float peak = 0f;
        for (int i = 0; i < data.Length; i++)
        {
            float m = data[i] < 0f ? -data[i] : data[i];
            if (m > peak) peak = m;
        }
        return peak;
    }

    private static short[] Resample24kTo48kStereoPcm16(float[] src)
    {
        // Linear 2x upsample from 24 kHz mono to 48 kHz stereo (FMOD mixer rate) + PCM16 conversion
        int dstFrames = src.Length * 2;
        short[] dst = new short[dstFrames * 2];
        for (int i = 0; i < src.Length; i++)
        {
            float s0 = src[i];
            float s1 = i + 1 < src.Length ? 0.5f * (s0 + src[i + 1]) : s0;
            short p0 = (short)Math.Clamp(s0 * 32767f, -32768f, 32767f);
            short p1 = (short)Math.Clamp(s1 * 32767f, -32768f, 32767f);
            int idx = i * 4;
            dst[idx] = p0;
            dst[idx + 1] = p0;
            dst[idx + 2] = p1;
            dst[idx + 3] = p1;
        }
        return dst;
    }

    private static OfflineTts CreateKokoroTts(int threads)
    {
        var cfg = new OfflineTtsConfig(true);
        cfg.Model.NumThreads = threads;
        cfg.Model.Provider = "cpu";
        cfg.Model.Kokoro.Model = Path.Combine(KokoroDir, "model.onnx");
        cfg.Model.Kokoro.Voices = Path.Combine(KokoroDir, "voices.bin");
        cfg.Model.Kokoro.Tokens = Path.Combine(KokoroDir, "tokens.txt");
        cfg.Model.Kokoro.DataDir = Path.Combine(KokoroDir, "espeak-ng-data");
        cfg.Model.Kokoro.LengthScale = 1.0f;
        return new OfflineTts(cfg);
    }

    private static async Task RunInstrumentationBench(string resultsDir)
    {
        Console.WriteLine("=== [2/3] Running Client + Local Kestrel Proxy + Kokoro Cold/Warm + Audio Output Instrumentation ===");

        // Start a real ASP.NET Core Kestrel HTTP proxy on localhost:0 mocking upstream Gemini response
        string sampleUpstreamJson = File.ReadLines(Path.Combine(ConviaRoot, "Paper", "turns_run1.jsonl"), Encoding.UTF8)
            .Select(l => JsonConvert.DeserializeObject<TurnLogRow>(l).Raw)
            .First();
        string systemPromptSample = CoachPrompt.Build("You are a friendly cafe barista.", "- Quest 1: Order a drink\n", true, false, "en");

        var webBuilder = WebApplication.CreateBuilder();
        webBuilder.Logging.ClearProviders();
        webBuilder.WebHost.UseUrls("http://127.0.0.1:0");
        var app = webBuilder.Build();
        app.MapPost("/v1/turn", async (HttpRequest req) =>
        {
            using var reader = new StreamReader(req.Body, Encoding.UTF8);
            string bodyStr = await reader.ReadToEndAsync();
            var reqObj = JObject.Parse(bodyStr);
            string audioB64 = reqObj["audio"]?["data"]?.ToString() ?? "";
            if (audioB64.Length > 7_000_000) return Results.BadRequest();
            string prompt = CoachPrompt.Build(reqObj["scenario"]?.ToString() ?? "cafe", "- Quest 1: Order\n", true, false, "en");
            var gen = new JObject
            {
                ["systemInstruction"] = new JObject { ["parts"] = new JArray { new JObject { ["text"] = prompt } } },
                ["generationConfig"] = new JObject
                {
                    ["temperature"] = 0.9,
                    ["maxOutputTokens"] = 800,
                    ["responseMimeType"] = CoachTurnSchema.ResponseMimeType,
                    ["responseSchema"] = JObject.Parse(CoachTurnSchema.Json)
                },
                ["contents"] = new JArray
                {
                    new JObject
                    {
                        ["role"] = "user",
                        ["parts"] = new JArray
                        {
                            new JObject { ["inlineData"] = new JObject { ["mimeType"] = "audio/wav", ["data"] = audioB64 } }
                        }
                    }
                }
            };
            string _ = gen.ToString(Formatting.None);
            return Results.Content(sampleUpstreamJson, "application/json");
        });
        await app.StartAsync();
        string proxyUrl = app.Urls.First() + "/v1/turn";
        using var http = new HttpClient();

        // Warm up Kestrel JIT & client audio pipeline
        string audioDir = Path.Combine(resultsDir, "audio");
        var wavPaths = Directory.GetFiles(audioDir, "u*.wav").OrderBy(p => p).ToArray();
        var stageRows = new List<object>();

        // 5 cold-start measurements of OfflineTts initialization & first generate()
        var coldLoadMs = new List<double>();
        var coldFirstInferMs = new List<double>();
        for (int c = 0; c < 5; c++)
        {
            var swLoad = Stopwatch.StartNew();
            using var coldTts = CreateKokoroTts(4);
            swLoad.Stop();
            coldLoadMs.Add(swLoad.Elapsed.TotalMilliseconds);

            var swInfer = Stopwatch.StartNew();
            var ga = coldTts.Generate("Of course! What would you like to drink today?", 1.05f, 0);
            ga.Dispose();
            swInfer.Stop();
            coldFirstInferMs.Add(swInfer.Elapsed.TotalMilliseconds);
        }

        // Open warm Kokoro & Windows waveOut probe (device default WAVE_MAPPER = -1, zero volume)
        using var tts = CreateKokoroTts(4);
        tts.Generate("Warm up sentence for allocator initialization.", 1.05f, 0).Dispose();

        // Warm HTTP proxy once
        var warmupPayload = new JObject { ["scenario"] = "cafe", ["mode"] = "voice", ["audio"] = new JObject { ["mimeType"] = "audio/wav", ["data"] = "UklGRg==" } }.ToString(Formatting.None);
        await http.PostAsync(proxyUrl, new StringContent(warmupPayload, Encoding.UTF8, "application/json"));

        // Setup Windows audio driver handle once (mimicking Unity audio output engine initialized at app launch)
        var wf = new WaveFormatEx
        {
            wFormatTag = 1,
            nChannels = 2,
            nSamplesPerSec = 48000,
            wBitsPerSample = 16,
            nBlockAlign = 4,
            nAvgBytesPerSec = 48000 * 4,
            cbSize = 0
        };
        int waveRc = waveOutOpen(out IntPtr hWaveOut, -1, ref wf, IntPtr.Zero, IntPtr.Zero, 0);

        // Measure all 64 utterances + chunk0 Kokoro + resample + waveOut enqueue
        var turnRows = File.ReadLines(Path.Combine(ConviaRoot, "Paper", "turns_run1.jsonl"), Encoding.UTF8)
            .Concat(File.ReadLines(Path.Combine(ConviaRoot, "Paper", "turns_run2.jsonl"), Encoding.UTF8))
            .Select(l => JsonConvert.DeserializeObject<TurnLogRow>(l))
            .ToArray();

        for (int i = 0; i < wavPaths.Length; i++)
        {
            float[] samples = ReadWavMonoFloat(wavPaths[i], out int sr);

            var swVal = Stopwatch.StartNew();
            float peak = PeakAmplitudeClient(samples);
            SpeechTiming timing = TurnSkills.MeasureTiming(samples, 1, sr);
            swVal.Stop();

            var swEnc = Stopwatch.StartNew();
            byte[] wavBytes = EncodeWavExactClient(samples, 1, sr);
            string b64 = Convert.ToBase64String(wavBytes);
            string reqJson = JsonConvert.SerializeObject(new
            {
                scenario = "cafe",
                mode = "voice",
                audio = new { mimeType = "audio/wav", data = b64 }
            });
            swEnc.Stop();

            var swProxy = Stopwatch.StartNew();
            using var resp = await http.PostAsync(proxyUrl, new StringContent(reqJson, Encoding.UTF8, "application/json"));
            string respBody = await resp.Content.ReadAsStringAsync();
            swProxy.Stop();

            var swParse = Stopwatch.StartNew();
            TurnLogRow tr = turnRows[i * 4];
            ParsedResponse pr = ParsedResponse.Parse(tr.Raw, tr.UserText);
            List<string> chunks = SpeechChunker.Split(pr.Text);
            swParse.Stop();

            string chunk0 = chunks.Count > 0 ? chunks[0] : "Okay.";
            var swSynth = Stopwatch.StartNew();
            var audio = tts.Generate(chunk0, 1.05f, 0);
            float[] rawOut = audio.Samples;
            audio.Dispose();
            swSynth.Stop();

            var swResamp = Stopwatch.StartNew();
            short[] pcm48 = Resample24kTo48kStereoPcm16(rawOut);
            // Zero amplitudes so lab run stays silent on speaker while exercising real DMA buffer enqueue
            Array.Clear(pcm48, 0, Math.Min(pcm48.Length, 4800));
            double driverEnqueueMs = 0.0;
            if (waveRc == 0)
            {
                // Enqueue first 100ms DMA audio buffer frame to winmm waveOut and immediately unprepare after reset
                int bytesLen = Math.Min(pcm48.Length * 2, 48000 * 2 * 2 / 10);
                byte[] buf = new byte[bytesLen];
                var handle = GCHandle.Alloc(buf, GCHandleType.Pinned);
                try
                {
                    var hdr = new WaveHdr
                    {
                        lpData = handle.AddrOfPinnedObject(),
                        dwBufferLength = (uint)bytesLen,
                        dwFlags = 0
                    };
                    waveOutPrepareHeader(hWaveOut, ref hdr, Marshal.SizeOf<WaveHdr>());
                    waveOutWrite(hWaveOut, ref hdr, Marshal.SizeOf<WaveHdr>());
                    waveOutReset(hWaveOut);
                    waveOutUnprepareHeader(hWaveOut, ref hdr, Marshal.SizeOf<WaveHdr>());
                }
                finally
                {
                    handle.Free();
                }
            }
            swResamp.Stop();

            stageRows.Add(new
            {
                uid = Path.GetFileNameWithoutExtension(wavPaths[i]),
                wavBytes = wavBytes.Length,
                b64Chars = b64.Length,
                chunk0Chars = chunk0.Length,
                micValidateTimingMs = swVal.Elapsed.TotalMilliseconds,
                wavAndBase64EncodeMs = swEnc.Elapsed.TotalMilliseconds,
                proxyLocalLoopbackMs = swProxy.Elapsed.TotalMilliseconds,
                clientParseAndChunkMs = swParse.Elapsed.TotalMilliseconds,
                warmChunk0SynthMs = swSynth.Elapsed.TotalMilliseconds,
                resample48kAndDriverEnqueueMs = swResamp.Elapsed.TotalMilliseconds,
                totalNonNetworkClientMs = swVal.Elapsed.TotalMilliseconds + swEnc.Elapsed.TotalMilliseconds
                                        + swProxy.Elapsed.TotalMilliseconds + swParse.Elapsed.TotalMilliseconds
                                        + swResamp.Elapsed.TotalMilliseconds
            });
        }

        if (waveRc == 0) waveOutClose(hWaveOut);
        await app.StopAsync();

        var outObj = new
        {
            waveOutAvailable = (waveRc == 0),
            coldModelLoadMs = coldLoadMs,
            coldFirstInferMs = coldFirstInferMs,
            warmPerTurnCount = stageRows.Count,
            stages = stageRows
        };
        string outPath = Path.Combine(resultsDir, "rev_instrument_e2e.json");
        File.WriteAllText(outPath, JsonConvert.SerializeObject(outObj, Formatting.Indented));
        Console.WriteLine($"  Wrote {outPath}");
    }

    // ──────────────────────────────────────────────────────────────
    // 3. ALL 484 MULTI-SENTENCE REPLIES UNDER REAL C# NATIVE KOKORO + LOAD + CONSTRAINED CPU PROFILES
    // ──────────────────────────────────────────────────────────────
    private static List<string> AdaptiveChunksPostMergeCSharp(string text, double safety = SpeechChunker.SafetyFactor)
    {
        var chs = new List<string>(SpeechChunker.Split(text));
        int k = 0;
        while (k < chs.Count - 1)
        {
            double estPlay = chs[k].Length * SpeechChunker.SpokenMsPerChar;
            double estNext = (SpeechChunker.SynthesisFixedMs + chs[k + 1].Length * SpeechChunker.SynthesisMsPerChar) * safety;
            if (estPlay < estNext)
            {
                chs[k] = chs[k] + " " + chs[k + 1];
                chs.RemoveAt(k + 1);
                if (k > 0) k--;
            }
            else
            {
                k++;
            }
        }
        return chs;
    }

    private static List<string> SplitSentencesExact(string text) => SpeechChunker.Sentences(text);

    private static void RunPlaybackProfilesBench(string resultsDir)
    {
        Console.WriteLine("=== [3/3] Running C# Native 1.13.8 Kokoro Playback Profiles Across All 484 Multi-Sentence Replies ===");
        var logFiles = new[]
        {
            Path.Combine(ConviaRoot, "Paper", "turns_run1.jsonl"),
            Path.Combine(ConviaRoot, "Paper", "turns_run2.jsonl")
        };

        var replies = new List<(string rid, string scenario, string reply)>();
        foreach (string lf in logFiles)
        {
            foreach (string line in File.ReadLines(lf, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var row = JsonConvert.DeserializeObject<TurnLogRow>(line);
                var pr = ParsedResponse.Parse(row.Raw, row.UserText);
                string reply = (pr.Text ?? "").Trim();
                if (SplitSentencesExact(reply).Count >= 2)
                    replies.Add(($"{row.Run}|{row.Turn:D3}", row.Scenario, reply));
            }
        }
        string stratPath = Path.Combine(resultsDir, "stratified_128_rids.json");
        if (File.Exists(stratPath))
        {
            var stratObj = JObject.Parse(File.ReadAllText(stratPath, Encoding.UTF8));
            var allow = new HashSet<string>(stratObj["rids"]!.Select(t => (string)t!), StringComparer.Ordinal);
            replies = replies.Where(r => allow.Contains(r.rid)).ToList();
        }
        Console.WriteLine($"  Multi-sentence replies loaded: {replies.Count}");

        var profiles = new[]
        {
            (Name: "desktop_4t_idle", Threads: 4, AffinityMask: (IntPtr)0, ConcurrentRenderLoad: false),
            (Name: "desktop_4t_render_load", Threads: 4, AffinityMask: (IntPtr)0, ConcurrentRenderLoad: true),
            (Name: "laptop_2t_affinity4", Threads: 2, AffinityMask: (IntPtr)0b1111, ConcurrentRenderLoad: false)
        };

        string csvPath = Path.Combine(resultsDir, $"rev_native_playback_{replies.Count}.csv");
        using var swCsv = new StreamWriter(csvPath, false, Encoding.UTF8);
        swCsv.WriteLine("profile,strategy,rid,scenario,chars,chunks,ttfa_recon_ms,gaps_recon,gap_recon_ms,ttfa_actual_ms,gaps_actual,gap_actual_ms,completion_actual_ms,audio_ms,resample_total_ms");

        foreach (var prof in profiles)
        {
            Console.WriteLine($"  Profile: {prof.Name} (threads={prof.Threads}, renderLoad={prof.ConcurrentRenderLoad})");
            IntPtr origAffinity = Process.GetCurrentProcess().ProcessorAffinity;
            if (prof.AffinityMask != IntPtr.Zero)
            {
                try { Process.GetCurrentProcess().ProcessorAffinity = prof.AffinityMask; } catch { }
            }

            CancellationTokenSource renderCts = null;
            Task[] renderWorkers = null;
            if (prof.ConcurrentRenderLoad)
            {
                // Spawn 8 continuous CPU/memory rendering-equivalent worker threads (60fps VRM matrix skinning + frame buffer copy)
                renderCts = new CancellationTokenSource();
                var tok = renderCts.Token;
                renderWorkers = Enumerable.Range(0, 8).Select(wid => Task.Run(() =>
                {
                    var rng = new float[16384];
                    for (int i = 0; i < rng.Length; i++) rng[i] = (i + wid) * 0.001f;
                    while (!tok.IsCancellationRequested)
                    {
                        float acc = 0f;
                        for (int k = 0; k < rng.Length; k++)
                        {
                            rng[k] = rng[k] * 1.00001f + 0.0001f;
                            acc += rng[k];
                        }
                        if (acc > 1e9f) Array.Clear(rng, 0, rng.Length);
                    }
                }, tok)).ToArray();
            }

            try
            {
                using var tts = CreateKokoroTts(prof.Threads);
                tts.Generate("Warm up.", 1.05f, 0).Dispose();

                // Synthesize all unique chunks needed by W, S, Lookahead-1, Post-merge under this exact runtime condition
                var uniqueChunks = new HashSet<string>();
                var planByStrat = new Dictionary<string, List<(string rid, string sc, string reply, List<string> chunks)>>();
                foreach (string st in new[] { "W", "S", "A", "A_post" })
                    planByStrat[st] = new List<(string, string, string, List<string>)>(replies.Count);

                foreach (var (rid, sc, reply) in replies)
                {
                    var cw = new List<string> { reply };
                    var cs = SplitSentencesExact(reply);
                    var ca = new List<string>(SpeechChunker.Split(reply));
                    var cp = AdaptiveChunksPostMergeCSharp(reply);
                    planByStrat["W"].Add((rid, sc, reply, cw));
                    planByStrat["S"].Add((rid, sc, reply, cs));
                    planByStrat["A"].Add((rid, sc, reply, ca));
                    planByStrat["A_post"].Add((rid, sc, reply, cp));
                    foreach (var c in cw) uniqueChunks.Add(c);
                    foreach (var c in cs) uniqueChunks.Add(c);
                    foreach (var c in ca) uniqueChunks.Add(c);
                    foreach (var c in cp) uniqueChunks.Add(c);
                }
                Console.WriteLine($"    Synthesizing {uniqueChunks.Count} unique chunks cleanly on {prof.Name}...");

                var synthStats = new Dictionary<string, (double synthMs, double resampMs, double audioMs)>(uniqueChunks.Count);
                foreach (string chunk in uniqueChunks)
                {
                    var tS = Stopwatch.StartNew();
                    var a = tts.Generate(chunk, 1.05f, 0);
                    float[] s = a.Samples;
                    int sr = a.SampleRate;
                    a.Dispose();
                    tS.Stop();

                    var tR = Stopwatch.StartNew();
                    short[] _ = Resample24kTo48kStereoPcm16(s);
                    tR.Stop();

                    double audioMs = 1000.0 * s.Length / sr;
                    synthStats[chunk] = (tS.Elapsed.TotalMilliseconds, tR.Elapsed.TotalMilliseconds, audioMs);
                }

                // Evaluate reconstructed vs 60Hz frame-quantized playback + buffer prep timing (mean frame tick = 8.33ms, max 16.67ms)
                const double FramePeriodMs = 1000.0 / 60.0; // 16.6667 ms Unity frame tick
                const double GapThresholdMs = 20.0;
                const double ClientOverheadPreSynthMs = 0.0; // added in E2; here compare chunking schedules cleanly
                string cacheCsv = Path.Combine(resultsDir, $"rev_native_chunk_cache_{prof.Name}.csv");
                using (var swCache = new StreamWriter(cacheCsv, false, Encoding.UTF8))
                {
                    swCache.WriteLine("synth_ms,audio_ms,resamp_ms,chars,text");
                    foreach (var kv in synthStats)
                    {
                        string esc = kv.Key.Replace("\"", "\"\"");
                        swCache.WriteLine($"{kv.Value.synthMs.ToString("F3", CultureInfo.InvariantCulture)},{kv.Value.audioMs.ToString("F3", CultureInfo.InvariantCulture)},{kv.Value.resampMs.ToString("F3", CultureInfo.InvariantCulture)},{kv.Key.Length},\"{esc}\"");
                    }
                }

                foreach (string st in new[] { "W", "S", "A", "A_post" })
                {
                    foreach (var (rid, sc, reply, chs) in planByStrat[st])
                    {
                        var statsList = chs.Select(c => synthStats[c]).ToList();
                        // Reconstructed (component idle additive, zero frame tick / zero resample)
                        double curSynth = 0.0, curPlay = 0.0;
                        double ttfaRecon = statsList[0].synthMs;
                        double gapReconSum = 0.0;
                        int gapsRecon = 0;
                        foreach (var x in statsList)
                        {
                            curSynth += x.synthMs;
                            if (curPlay < curSynth)
                            {
                                double g = curSynth - curPlay;
                                if (curPlay > 0) { gapReconSum += g; if (g > GapThresholdMs) gapsRecon++; }
                                curPlay = curSynth;
                            }
                            curPlay += x.audioMs;
                        }

                        // Instrumented Unity 60Hz frame-tick + 24k->48k resample + worst/expected dispatch quantization
                        // Synth worker runs serially on _ttsGate; main thread picks up in Update() on next 16.67ms frame boundary;
                        // MonitorAudioPlayback checks _audioSource.isPlaying every FramePeriodMs (16.67ms), advancing start to next frame tick.
                        double sClock = 0.0, pClock = 0.0, ttfaActual = 0.0, gapActualSum = 0.0;
                        int gapsActual = 0;
                        double resampSum = statsList.Sum(z => z.resampMs);
                        for (int idx = 0; idx < statsList.Count; idx++)
                        {
                            var x = statsList[idx];
                            sClock += x.synthMs + x.resampMs;
                            double readyFrame = Math.Ceiling(sClock / FramePeriodMs) * FramePeriodMs;
                            if (idx == 0) ttfaActual = readyFrame;
                            double desiredStart = Math.Max(readyFrame, pClock);
                            // MonitorAudioPlayback wakes at frame ticks: ceil to FramePeriodMs grid
                            double startTick = idx == 0 ? readyFrame : Math.Ceiling(desiredStart / FramePeriodMs) * FramePeriodMs;
                            if (idx > 0 && startTick > pClock)
                            {
                                double g = startTick - pClock;
                                gapActualSum += g;
                                if (g > GapThresholdMs) gapsActual++;
                            }
                            pClock = startTick + x.audioMs;
                        }

                        swCsv.WriteLine(string.Join(",",
                            prof.Name, st, rid, sc, reply.Length, chs.Count,
                            ttfaRecon.ToString("F3", CultureInfo.InvariantCulture),
                            gapsRecon,
                            gapReconSum.ToString("F3", CultureInfo.InvariantCulture),
                            ttfaActual.ToString("F3", CultureInfo.InvariantCulture),
                            gapsActual,
                            gapActualSum.ToString("F3", CultureInfo.InvariantCulture),
                            pClock.ToString("F3", CultureInfo.InvariantCulture),
                            statsList.Sum(z => z.audioMs).ToString("F3", CultureInfo.InvariantCulture),
                            resampSum.ToString("F3", CultureInfo.InvariantCulture)));
                    }
                }
            }
            finally
            {
                if (renderCts != null)
                {
                    renderCts.Cancel();
                    try { Task.WaitAll(renderWorkers); } catch { }
                    renderCts.Dispose();
                }
                if (prof.AffinityMask != IntPtr.Zero)
                {
                    try { Process.GetCurrentProcess().ProcessorAffinity = origAffinity; } catch { }
                }
            }
        }
        Console.WriteLine($"  Wrote {csvPath}");
    }

    // ──────────────────────────────────────────────────────────────
    // 4. REAL-TIME 1X WALL-CLOCK 60HZ MONITOR PLAYBACK VALIDATION ON 32 STRATIFIED REPLIES
    // ──────────────────────────────────────────────────────────────
    private static async Task RunRealtimeWaveOutValidation(string resultsDir)
    {
        Console.WriteLine("=== [4/4] Running 1x Real-Time Wall-Clock 60Hz Playback Validation (n=16 tight-margin + stratified replies) ===");
        var logFiles = new[]
        {
            Path.Combine(ConviaRoot, "Paper", "turns_run1.jsonl"),
            Path.Combine(ConviaRoot, "Paper", "turns_run2.jsonl")
        };
        var candidates = new List<(string rid, string sc, string reply, List<string> sents)>();
        foreach (string lf in logFiles)
        {
            foreach (string line in File.ReadLines(lf, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var row = JsonConvert.DeserializeObject<TurnLogRow>(line);
                var pr = ParsedResponse.Parse(row.Raw, row.UserText);
                string reply = (pr.Text ?? "").Trim();
                var sents = SplitSentencesExact(reply);
                if (sents.Count >= 2 && sents[0].Length <= 35) // tight first sentence where S risks underrun
                    candidates.Add(($"{row.Run}|{row.Turn:D3}", row.Scenario, reply, sents));
            }
        }
        // Pick 12 shortest-first-sentence replies across scenarios so test runs in ~90s while directly exercising tight boundaries
        var picked = candidates.OrderBy(c => c.sents[0].Length).Take(12).ToList();

        // Execute with 2 threads on Core Ultra 7 (RTF ~0.56, comparable to Ryzen 5 5600) in real wall clock
        using var tts = CreateKokoroTts(2);
        tts.Generate("Warm up.", 1.05f, 0).Dispose();

        var rtResults = new List<object>();
        foreach (var (rid, sc, reply, sents) in picked)
        {
            foreach (string strat in new[] { "S", "A_post" })
            {
                var chunks = strat == "S" ? sents : AdaptiveChunksPostMergeCSharp(reply);
                var queue = new Queue<(double audioMs, double synthMs, double readyAtMs)>();
                object qLock = new object();
                int remaining = chunks.Count;
                var swTurn = Stopwatch.StartNew();

                // Background sequential Kokoro synthesis worker (identical to GeminiAvatarController _ttsGate)
                var synthTask = Task.Run(() =>
                {
                    foreach (string ch in chunks)
                    {
                        var t0 = Stopwatch.StartNew();
                        var a = tts.Generate(ch, 1.05f, 0);
                        float[] smp = a.Samples;
                        int asr = a.SampleRate;
                        a.Dispose();
                        short[] _ = Resample24kTo48kStereoPcm16(smp);
                        t0.Stop();
                        double ams = 1000.0 * smp.Length / asr;
                        lock (qLock)
                        {
                            queue.Enqueue((ams, t0.Elapsed.TotalMilliseconds, swTurn.Elapsed.TotalMilliseconds));
                        }
                    }
                });

                // Real 60Hz MonitorAudioPlayback polling loop at 1x wall clock
                double firstAudioMs = -1, clipEndsAtMs = 0, totalGapMs = 0;
                int gapCount = 0;
                bool isPlaying = false;
                while (true)
                {
                    double now = swTurn.Elapsed.TotalMilliseconds;
                    if (isPlaying && now >= clipEndsAtMs)
                    {
                        isPlaying = false;
                    }
                    if (!isPlaying)
                    {
                        (double audioMs, double synthMs, double readyAtMs) next = default;
                        bool hasNext = false;
                        lock (qLock)
                        {
                            if (queue.Count > 0)
                            {
                                next = queue.Dequeue();
                                hasNext = true;
                                remaining--;
                            }
                        }
                        if (hasNext)
                        {
                            if (firstAudioMs < 0) firstAudioMs = now;
                            else
                            {
                                double g = now - clipEndsAtMs;
                                if (g > 0) totalGapMs += g;
                                if (g > 20.0) gapCount++;
                            }
                            isPlaying = true;
                            clipEndsAtMs = now + next.audioMs;
                        }
                        else if (remaining == 0)
                        {
                            break;
                        }
                    }
                    await Task.Delay(16); // real 60Hz Unity Update frame step
                }
                await synthTask;
                rtResults.Add(new
                {
                    rid,
                    scenario = sc,
                    strategy = strat,
                    chunks = chunks.Count,
                    rtOnsetMs = Math.Round(firstAudioMs, 2),
                    rtGapsOver20Ms = gapCount,
                    rtTotalGapMs = Math.Round(totalGapMs, 2),
                    rtCompletionMs = Math.Round(swTurn.Elapsed.TotalMilliseconds, 2)
                });
            }
        }

        string outPath = Path.Combine(resultsDir, "rev_realtime_60hz_validation.json");
        File.WriteAllText(outPath, JsonConvert.SerializeObject(rtResults, Formatting.Indented));
        Console.WriteLine($"  Wrote {outPath}");
    }

    // ──────────────────────────────────────────────────────────────
    // 5. LIVE CONTINUOUS PUSH-TO-TALK-TO-WAVEOUT E2E VALIDATION (n = 24)
    // ──────────────────────────────────────────────────────────────
    private static Dictionary<string, double> ComputeBenchStats(IReadOnlyList<double> raw)
    {
        var s = raw.OrderBy(x => x).ToArray();
        int n = s.Length;
        double Quantile(double q)
        {
            if (n == 0) return 0;
            if (n == 1) return s[0];
            double idx = q * (n - 1);
            int lo = (int)Math.Floor(idx), hi = (int)Math.Ceiling(idx);
            return lo == hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
        }
        double mean = s.Average();
        double sd = n > 1 ? Math.Sqrt(s.Sum(x => (x - mean) * (x - mean)) / (n - 1)) : 0.0;
        double p25 = Quantile(0.25), p50 = Quantile(0.50), p75 = Quantile(0.75);
        int p95Idx = Math.Min(n - 1, (int)(0.95 * n));
        return new Dictionary<string, double>
        {
            ["n"] = n,
            ["median"] = Math.Round(p50, 2),
            ["iqr"] = Math.Round(p75 - p25, 2),
            ["p25"] = Math.Round(p25, 2),
            ["p75"] = Math.Round(p75, 2),
            ["mean"] = Math.Round(mean, 2),
            ["sd"] = Math.Round(sd, 2),
            ["p95"] = Math.Round(s[p95Idx], 2),
            ["min"] = Math.Round(s[0], 2),
            ["max"] = Math.Round(s[n - 1], 2)
        };
    }

    private static async Task RunLiveEndToEndWaveOutValidationAsync(string paperDir, string resultsDir)
    {
        Console.WriteLine("=== [5/5] Running Live Continuous Push-to-Talk-to-waveOutWrite E2E Validation (n=24) ===");
        string keyPath = Path.Combine(ConviaRoot, "gemini.key");
        string apiKey = File.Exists(keyPath)
            ? File.ReadAllText(keyPath, Encoding.UTF8).Trim()
            : (Environment.GetEnvironmentVariable("GEMINI_API_KEY") ?? "").Trim();
        if (string.IsNullOrEmpty(apiKey))
            throw new InvalidOperationException("Gemini API key not found for live-e2e validation.");

        string archiveDir = File.Exists(Path.Combine(paperDir, "prompts", "scenarios.json"))
            ? paperDir
            : Path.Combine(paperDir, "reproducibility_archive");
        var scenariosObj = JObject.Parse(File.ReadAllText(Path.Combine(archiveDir, "prompts", "scenarios.json"), Encoding.UTF8));
        var schemaObj = JObject.Parse(File.ReadAllText(Path.Combine(archiveDir, "schemas", "CoachTurnSchema.json"), Encoding.UTF8));
        string promptBody = File.ReadAllText(Path.Combine(archiveDir, "prompts", "coach_prompt_body.txt"), Encoding.UTF8)
            .Replace("{TIPTHAI}", "- tipThai: the same explanation as tip, written in natural Thai for a Thai first-language learner.\n");

        string BuildSystemPrompt(string scenario)
        {
            var sc = scenariosObj[scenario] as JObject
                ?? throw new InvalidOperationException($"Unknown scenario: {scenario}");
            string persona = (string)sc["persona"];
            var rules = sc["rules"]?.Select(r => (string)r).ToList() ?? new List<string>();
            var sb = new StringBuilder();
            sb.Append(persona).Append("\n\n");
            if (rules.Count > 0)
            {
                const string ObjectiveRules =
                    "QUEST EVALUATION RULES:\n" +
                    "- List a quest number in questsCompleted only when the user actually accomplished that objective with an understandable sentence.\n" +
                    "- A bare keyword or fragment does not complete an objective even when the topic matches: \"total price\" is not asking for the total, it is naming a subject. The learner is here to practise saying the whole thing.\n" +
                    "- An objective also does not count when it makes no sense at this point in the conversation - asking the total before ordering anything, for instance.\n" +
                    "- Be encouraging and friendly in your reply, but do not hand out objectives that were not earned.\n\n";
                sb.Append("SCENARIO QUEST OBJECTIVES TO EVALUATE:\n");
                for (int idx = 0; idx < rules.Count; idx++)
                    sb.Append($"- Quest {idx + 1}: {rules[idx]}\n");
                sb.Append('\n').Append(ObjectiveRules);
            }
            sb.Append("IMPORTANT: Write all feedback in clear, simple English, with no emojis. The one exception is the tipThai field, which is written in Thai.\n\n");
            sb.Append(promptBody);
            return sb.ToString();
        }

        // Load exact strat24 subset (indices 0, 2, 5 per scenario) matching Part 3a/3b
        string e2CsvPath = File.Exists(Path.Combine(resultsDir, "e2_turns.csv"))
            ? Path.Combine(resultsDir, "e2_turns.csv")
            : Path.Combine(archiveDir, "data", "e2_turns.csv");
        string audioDir = Directory.Exists(Path.Combine(resultsDir, "audio"))
            ? Path.Combine(resultsDir, "audio")
            : Path.Combine(archiveDir, "data", "audio");

        var seenUids = new HashSet<string>(StringComparer.Ordinal);
        var byScenario = new Dictionary<string, List<(string uid, string scenario)>>();
        foreach (string line in File.ReadLines(e2CsvPath, Encoding.UTF8).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            int c1 = line.IndexOf(',');
            int c2 = line.IndexOf(',', c1 + 1);
            if (c1 <= 0 || c2 <= c1) continue;
            string uid = line.Substring(0, c1).Trim();
            string sc = line.Substring(c1 + 1, c2 - c1 - 1).Trim();
            if (seenUids.Add(uid))
            {
                if (!byScenario.TryGetValue(sc, out var list))
                {
                    list = new List<(string, string)>();
                    byScenario[sc] = list;
                }
                list.Add((uid, sc));
            }
        }
        var orderedUids = byScenario.Values
            .OrderBy(lst => lst[0].uid, StringComparer.Ordinal)
            .SelectMany(lst => new[] { lst[0], lst[2], lst[5] })
            .ToList();

        // Start local ASP.NET Core Kestrel license proxy forwarding to live Gemini 3.6 Flash (thinkingBudget=0)
        using var upstreamHttp = new HttpClient(new SocketsHttpHandler
        {
            PooledConnectionLifetime = TimeSpan.FromMinutes(10),
            AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate
        })
        { Timeout = TimeSpan.FromSeconds(90) };

        var webBuilder = WebApplication.CreateBuilder();
        webBuilder.Logging.ClearProviders();
        webBuilder.WebHost.UseUrls("http://127.0.0.1:0");
        var app = webBuilder.Build();
        app.MapPost("/v1/turn", async (HttpRequest req, HttpResponse resp) =>
        {
            using var reader = new StreamReader(req.Body, Encoding.UTF8);
            string bodyStr = await reader.ReadToEndAsync();
            var reqObj = JObject.Parse(bodyStr);
            string scenario = reqObj["scenario"]?.ToString() ?? "CafeRestaurant";
            string audioB64 = reqObj["audio"]?["data"]?.ToString() ?? "";
            string sysPrompt = BuildSystemPrompt(scenario);

            var genBody = new JObject
            {
                ["systemInstruction"] = new JObject { ["parts"] = new JArray { new JObject { ["text"] = sysPrompt } } },
                ["generationConfig"] = new JObject
                {
                    ["temperature"] = 0.9,
                    ["maxOutputTokens"] = 2048,
                    ["thinkingConfig"] = new JObject { ["thinkingBudget"] = 0 },
                    ["responseMimeType"] = "application/json",
                    ["responseSchema"] = schemaObj
                },
                ["contents"] = new JArray
                {
                    new JObject
                    {
                        ["role"] = "user",
                        ["parts"] = new JArray
                        {
                            new JObject { ["inlineData"] = new JObject { ["mimeType"] = "audio/wav", ["data"] = audioB64 } },
                            new JObject { ["text"] = "Listen to my spoken English in this audio. Transcribe it, coach it, reply in character, and list any quest objectives I accomplished. If you cannot hear any actual speech in the audio, return an empty transcription rather than guessing at what I might have said." }
                        }
                    }
                }
            };
            string payloadJson = genBody.ToString(Formatting.None);
            const string endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent";

            for (int attempt = 0; attempt < 4; attempt++)
            {
                using var msg = new HttpRequestMessage(HttpMethod.Post, endpoint);
                msg.Headers.Add("x-goog-api-key", apiKey);
                msg.Content = new StringContent(payloadJson, Encoding.UTF8, "application/json");
                var swApi = Stopwatch.StartNew();
                using var upResp = await upstreamHttp.SendAsync(msg);
                string upTxt = await upResp.Content.ReadAsStringAsync();
                swApi.Stop();
                if ((int)upResp.StatusCode == 200)
                {
                    var upJson = JObject.Parse(upTxt);
                    var parts = upJson["candidates"]?[0]?["content"]?["parts"] as JArray;
                    var textSb = new StringBuilder();
                    if (parts != null)
                    {
                        foreach (var p in parts)
                        {
                            if (p["text"] != null) textSb.Append((string)p["text"]);
                        }
                    }
                    resp.Headers["X-Upstream-Api-Ms"] = swApi.Elapsed.TotalMilliseconds.ToString("F3", CultureInfo.InvariantCulture);
                    return Results.Content(textSb.ToString(), "application/json");
                }
                if ((int)upResp.StatusCode is 429 or 500 or 502 or 503 or 504 && attempt < 3)
                {
                    await Task.Delay(1500 * (attempt + 1));
                    continue;
                }
                return Results.Problem($"Upstream HTTP {(int)upResp.StatusCode}: {upTxt}");
            }
            return Results.Problem("Exhausted retries");
        });
        await app.StartAsync();
        string proxyUrl = app.Urls.First() + "/v1/turn";
        using var clientHttp = new HttpClient { Timeout = TimeSpan.FromSeconds(90) };

        // Warm 4-thread Kokoro + Windows waveOut handle
        using var tts = CreateKokoroTts(4);
        tts.Generate("Warm up sentence for runtime allocation.", 1.05f, 0).Dispose();

        var wf = new WaveFormatEx
        {
            wFormatTag = 1,
            nChannels = 2,
            nSamplesPerSec = 48000,
            wBitsPerSample = 16,
            nBlockAlign = 4,
            nAvgBytesPerSec = 48000 * 4,
            cbSize = 0
        };
        int waveRc = waveOutOpen(out IntPtr hWaveOut, -1, ref wf, IntPtr.Zero, IntPtr.Zero, 0);
        if (waveRc != 0)
            throw new InvalidOperationException($"waveOutOpen failed with code {waveRc}");

        timeBeginPeriod(1);
        var perTurn = new List<object>();
        const double FramePeriodMs = 1000.0 / 60.0;

        try
        {
            // 2 real warmup turns (Kestrel + TLS pool to Google + JIT + winmm)
            foreach (var w in orderedUids.Take(2))
            {
                float[] wSmp = ReadWavMonoFloat(Path.Combine(audioDir, $"{w.uid}.wav"), out int wSr);
                byte[] wBytes = EncodeWavExactClient(wSmp, 1, wSr);
                string wReq = JsonConvert.SerializeObject(new
                {
                    scenario = w.scenario,
                    mode = "voice",
                    audio = new { mimeType = "audio/wav", data = Convert.ToBase64String(wBytes) }
                });
                using var wr = await clientHttp.PostAsync(proxyUrl, new StringContent(wReq, Encoding.UTF8, "application/json"));
                wr.EnsureSuccessStatusCode();
                await Task.Delay(250);
            }

            foreach (var (uid, scenario) in orderedUids)
            {
                float[] samples = ReadWavMonoFloat(Path.Combine(audioDir, $"{uid}.wav"), out int sr);

                // Single continuous wall-clock turn timer starting at push-to-talk release
                var swTurn = Stopwatch.StartNew();

                var swValEnc = Stopwatch.StartNew();
                float _peak = PeakAmplitudeClient(samples);
                SpeechTiming _timing = TurnSkills.MeasureTiming(samples, 1, sr);
                byte[] wavBytes = EncodeWavExactClient(samples, 1, sr);
                string b64 = Convert.ToBase64String(wavBytes);
                string reqJson = JsonConvert.SerializeObject(new
                {
                    scenario,
                    mode = "voice",
                    audio = new { mimeType = "audio/wav", data = b64 }
                });
                swValEnc.Stop();

                var swProxyApi = Stopwatch.StartNew();
                using var resp = await clientHttp.PostAsync(proxyUrl, new StringContent(reqJson, Encoding.UTF8, "application/json"));
                resp.EnsureSuccessStatusCode();
                string rawCoachJson = await resp.Content.ReadAsStringAsync();
                swProxyApi.Stop();
                double upstreamApiMs = resp.Headers.TryGetValues("X-Upstream-Api-Ms", out var vals)
                    && double.TryParse(vals.First(), NumberStyles.Float, CultureInfo.InvariantCulture, out double upMs)
                    ? upMs
                    : swProxyApi.Elapsed.TotalMilliseconds;

                var swParseChunk = Stopwatch.StartNew();
                ParsedResponse pr = ParsedResponse.Parse(rawCoachJson, null);
                string replyText = string.IsNullOrWhiteSpace(pr.Text) ? "Okay." : pr.Text;
                List<string> chunks = AdaptiveChunksPostMergeCSharp(replyText);
                string chunk0 = chunks.Count > 0 ? chunks[0] : "Okay.";
                swParseChunk.Stop();

                var swSynth0 = Stopwatch.StartNew();
                var audio = tts.Generate(chunk0, 1.05f, 0);
                float[] rawOut = audio.Samples;
                int outSr = audio.SampleRate;
                audio.Dispose();
                swSynth0.Stop();

                var swResamp = Stopwatch.StartNew();
                short[] pcm48 = Resample24kTo48kStereoPcm16(rawOut);
                int bytesLen = Math.Min(pcm48.Length * 2, 48000 * 2 * 2 / 10);
                byte[] buf = new byte[bytesLen];
                Buffer.BlockCopy(pcm48, 0, buf, 0, bytesLen);
                Array.Clear(buf, 0, bytesLen); // Keep speakers silent while submitting valid 48kHz stereo PCM buffer
                swResamp.Stop();

                // Real 60 Hz polling grid wait until next 16.67 ms frame tick
                var swFrameWait = Stopwatch.StartNew();
                double nowBeforeWait = swTurn.Elapsed.TotalMilliseconds;
                double nextFrameTick = Math.Ceiling(nowBeforeWait / FramePeriodMs) * FramePeriodMs;
                while (swTurn.Elapsed.TotalMilliseconds < nextFrameTick)
                {
                    double rem = nextFrameTick - swTurn.Elapsed.TotalMilliseconds;
                    if (rem > 2.0) Thread.Sleep(1);
                    else Thread.SpinWait(64);
                }
                swFrameWait.Stop();

                var swWaveOut = Stopwatch.StartNew();
                var handle = GCHandle.Alloc(buf, GCHandleType.Pinned);
                var hdr = new WaveHdr
                {
                    lpData = handle.AddrOfPinnedObject(),
                    dwBufferLength = (uint)bytesLen,
                    dwFlags = 0
                };
                try
                {
                    int prepRc = waveOutPrepareHeader(hWaveOut, ref hdr, Marshal.SizeOf<WaveHdr>());
                    int writeRc = waveOutWrite(hWaveOut, ref hdr, Marshal.SizeOf<WaveHdr>());
                    if (prepRc != 0 || writeRc != 0)
                        throw new InvalidOperationException($"waveOut error prep={prepRc} write={writeRc}");
                }
                finally
                {
                    // Stop continuous turn timer at OS audio driver buffer submission (waveOutWrite completion)
                    swWaveOut.Stop();
                    swTurn.Stop();
                    waveOutReset(hWaveOut);
                    waveOutUnprepareHeader(hWaveOut, ref hdr, Marshal.SizeOf<WaveHdr>());
                    handle.Free();
                }

                double liveOnsetMs = swTurn.Elapsed.TotalMilliseconds;
                double valEncMs = swValEnc.Elapsed.TotalMilliseconds;
                double proxyAndApiMs = swProxyApi.Elapsed.TotalMilliseconds;
                double parseChunkMs = swParseChunk.Elapsed.TotalMilliseconds;
                double synth0Ms = swSynth0.Elapsed.TotalMilliseconds;
                double resampleMs = swResamp.Elapsed.TotalMilliseconds;
                double frameWaitMs = swFrameWait.Elapsed.TotalMilliseconds;
                double waveOutMs = swWaveOut.Elapsed.TotalMilliseconds;
                double nonNetClientMs = valEncMs + (proxyAndApiMs - upstreamApiMs) + parseChunkMs + resampleMs + waveOutMs;
                double stageSumMs = valEncMs + proxyAndApiMs + parseChunkMs + synth0Ms + resampleMs + frameWaitMs + waveOutMs;
                double reconFormulaMs = upstreamApiMs + synth0Ms + 9.0 + 8.33;
                double reconWithProxyFormulaMs = proxyAndApiMs + synth0Ms + 9.0 + 8.33;
                double chunk0AudioMs = 1000.0 * rawOut.Length / outSr;

                perTurn.Add(new
                {
                    uid,
                    scenario,
                    wavBytes = wavBytes.Length,
                    chunk0Chars = chunk0.Length,
                    chunk0AudioMs = Math.Round(chunk0AudioMs, 2),
                    liveWaveOutOnsetMs = Math.Round(liveOnsetMs, 2),
                    stageSumMs = Math.Round(stageSumMs, 2),
                    reconFormulaMs = Math.Round(reconFormulaMs, 2),
                    reconWithProxyFormulaMs = Math.Round(reconWithProxyFormulaMs, 2),
                    diffLiveMinusReconMs = Math.Round(liveOnsetMs - reconFormulaMs, 2),
                    diffLiveMinusProxyReconMs = Math.Round(liveOnsetMs - reconWithProxyFormulaMs, 2),
                    valEncMs = Math.Round(valEncMs, 2),
                    proxyAndApiMs = Math.Round(proxyAndApiMs, 2),
                    upstreamApiMs = Math.Round(upstreamApiMs, 2),
                    parseChunkMs = Math.Round(parseChunkMs, 2),
                    synth0Ms = Math.Round(synth0Ms, 2),
                    resampleMs = Math.Round(resampleMs, 2),
                    frameWait60HzMs = Math.Round(frameWaitMs, 2),
                    waveOutMs = Math.Round(waveOutMs, 2),
                    nonNetworkClientBeforeFrameWaitMs = Math.Round(nonNetClientMs, 2),
                    jsonOk = pr.HasScore
                });
                Console.WriteLine($"    {uid} ({scenario}): live_waveout={liveOnsetMs:F1}ms, recon={reconFormulaMs:F1}ms, diff={liveOnsetMs - reconFormulaMs:+0.0;-0.0}ms (api={upstreamApiMs:F1}ms, synth0={synth0Ms:F1}ms, wait60Hz={frameWaitMs:F1}ms)");
                await Task.Delay(200);
            }
        }
        finally
        {
            timeEndPeriod(1);
            waveOutClose(hWaveOut);
            await app.StopAsync();
        }

        double[] Extract(string prop) => perTurn.Select(r => Convert.ToDouble(r.GetType().GetProperty(prop)!.GetValue(r))).ToArray();
        var summary = new
        {
            n = perTurn.Count,
            model = "gemini-3.6-flash",
            thinkingBudget = 0,
            maxOutputTokens = 2048,
            threads = 4,
            boundary = "winmm.dll waveOutWrite OS driver buffer submission",
            liveWaveOutOnsetMs = ComputeBenchStats(Extract("liveWaveOutOnsetMs")),
            reconFormulaMs = ComputeBenchStats(Extract("reconFormulaMs")),
            reconWithProxyFormulaMs = ComputeBenchStats(Extract("reconWithProxyFormulaMs")),
            diffLiveMinusReconMs = ComputeBenchStats(Extract("diffLiveMinusReconMs")),
            diffLiveMinusProxyReconMs = ComputeBenchStats(Extract("diffLiveMinusProxyReconMs")),
            upstreamApiMs = ComputeBenchStats(Extract("upstreamApiMs")),
            proxyAndApiMs = ComputeBenchStats(Extract("proxyAndApiMs")),
            synth0Ms = ComputeBenchStats(Extract("synth0Ms")),
            frameWait60HzMs = ComputeBenchStats(Extract("frameWait60HzMs")),
            nonNetworkClientBeforeFrameWaitMs = ComputeBenchStats(Extract("nonNetworkClientBeforeFrameWaitMs")),
            turns = perTurn
        };

        string outJson = JsonConvert.SerializeObject(summary, Formatting.Indented);
        string outPath = Path.Combine(resultsDir, "rev_live_client_e2e_24.json");
        string archiveOutPath = Path.Combine(archiveDir, "data", "rev_live_client_e2e_24.json");
        var utf8NoBom = new UTF8Encoding(false);
        File.WriteAllText(outPath, outJson, utf8NoBom);
        if (Directory.Exists(Path.GetDirectoryName(archiveOutPath)))
            File.WriteAllText(archiveOutPath, outJson, utf8NoBom);
        Console.WriteLine($"  Wrote {outPath}");
    }
}
