using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace GeminiAvatar
{
    /// <summary>
    ///     Records one line per completed turn so that the model's behaviour can be measured
    ///     rather than asserted.
    ///
    ///     <para>
    ///     Three things are captured that the application otherwise discards. The raw payload,
    ///     recorded <i>before</i> <see cref="ParsedResponse.Parse"/> runs, so the enforcement
    ///     layer can be replayed offline against what the model actually returned. The token
    ///     counts the service reports in <c>usageMetadata</c>, which arrive on every response
    ///     and were previously parsed and dropped. And the wall-clock time the request took.
    ///     </para>
    ///
    ///     <para>
    ///     The format is JSON Lines: one self-contained object per line, appended, so a run that
    ///     is interrupted still leaves valid data and a long run does not have to be held in
    ///     memory. Strings are escaped by hand rather than through a serializer because the raw
    ///     payload is arbitrary model output and must survive verbatim.
    ///     </para>
    ///
    ///     <para>
    ///     Off by default. Turning it on writes learner utterances and model output to disk, so
    ///     it is a study instrument and not something to ship enabled.
    ///     </para>
    /// </summary>
    public static class TurnTelemetry
    {
        /// <summary>Set true to record. Left false so a normal build writes nothing.</summary>
        public static bool Enabled;

        /// <summary>Free-form label written into every line, e.g. the run identifier.</summary>
        public static string RunLabel = "";

        private static string _path;
        private static int _turnIndex;

        /// <summary>File the current run is appending to. Empty until the first write.</summary>
        public static string CurrentPath => _path ?? "";

        /// <summary>Turns recorded in this run.</summary>
        public static int TurnCount => _turnIndex;

        /// <summary>
        ///     Starts a new run. Existing files are never overwritten: the name carries a
        ///     timestamp, so an accidental second start costs a file rather than the data.
        /// </summary>
        public static void BeginRun(string label)
        {
            RunLabel = label ?? "";
            _turnIndex = 0;

            string folder = Path.Combine(Application.persistentDataPath, "TurnTelemetry");
            Directory.CreateDirectory(folder);

            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            _path = Path.Combine(folder, $"turns_{stamp}.jsonl");

            Debug.Log($"[Telemetry] Recording to {_path}");
        }

        /// <summary>
        ///     Appends one turn.
        /// </summary>
        /// <param name="scenario">Practice situation in force.</param>
        /// <param name="mode">"voice" or "text", so audio and typed turns can be separated.</param>
        /// <param name="userText">What the learner said or typed. Empty for a voice turn where
        ///     the transcript is only known from the response.</param>
        /// <param name="rawPayload">The model's output verbatim, before any parsing.</param>
        /// <param name="usage">Token counts as reported by the service, or null.</param>
        /// <param name="requestMs">Wall-clock milliseconds from request sent to response received.</param>
        /// <param name="audioBytes">Size of the audio sent, or 0 for a typed turn.</param>
        /// <param name="model">The model that answered, as the service reported it.</param>
        public static void Record(PracticeScenario scenario, string mode, string userText,
                                  string rawPayload, GeminiUsageMetadata usage,
                                  double requestMs, int audioBytes, string model)
        {
            if (!Enabled) return;

            if (_path == null)
                BeginRun(RunLabel);

            _turnIndex++;

            var sb = new StringBuilder(rawPayload != null ? rawPayload.Length + 512 : 512);
            sb.Append('{');
            Field(sb, "run", RunLabel); sb.Append(',');
            Field(sb, "turn", _turnIndex); sb.Append(',');
            Field(sb, "at", DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture)); sb.Append(',');
            Field(sb, "scenario", scenario.ToString()); sb.Append(',');
            Field(sb, "mode", mode); sb.Append(',');
            Field(sb, "model", model ?? ""); sb.Append(',');
            Field(sb, "userText", userText ?? ""); sb.Append(',');
            Field(sb, "audioBytes", audioBytes); sb.Append(',');
            Field(sb, "requestMs", Math.Round(requestMs, 1)); sb.Append(',');
            Field(sb, "promptTokens", usage != null ? usage.promptTokenCount : -1); sb.Append(',');
            Field(sb, "outputTokens", usage != null ? usage.candidatesTokenCount : -1); sb.Append(',');
            Field(sb, "thoughtTokens", usage != null ? usage.thoughtsTokenCount : -1); sb.Append(',');
            Field(sb, "totalTokens", usage != null ? usage.totalTokenCount : -1); sb.Append(',');
            Field(sb, "raw", rawPayload ?? "");
            sb.Append('}');

            try
            {
                // One append per turn: a run that crashes still leaves every completed turn.
                File.AppendAllText(_path, sb.ToString() + "\n", new UTF8Encoding(false));
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[Telemetry] Could not append turn {_turnIndex}: {e.Message}");
            }
        }

        private static void Field(StringBuilder sb, string key, string value)
        {
            sb.Append('"').Append(key).Append("\":");
            AppendJsonString(sb, value);
        }

        private static void Field(StringBuilder sb, string key, int value)
        {
            sb.Append('"').Append(key).Append("\":")
              .Append(value.ToString(CultureInfo.InvariantCulture));
        }

        private static void Field(StringBuilder sb, string key, double value)
        {
            sb.Append('"').Append(key).Append("\":")
              .Append(value.ToString("0.#", CultureInfo.InvariantCulture));
        }

        /// <summary>
        ///     Minimal JSON string escaping. Hand-written because the payload is arbitrary model
        ///     output, including Thai text and embedded newlines, and must round-trip byte for
        ///     byte through the analysis step.
        /// </summary>
        private static void AppendJsonString(StringBuilder sb, string s)
        {
            sb.Append('"');
            foreach (char c in s ?? "")
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    default:
                        if (c < 0x20)
                            sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else
                            sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }
    }
}
