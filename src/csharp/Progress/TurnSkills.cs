using System;
using System.Collections.Generic;

namespace GeminiAvatar
{
    /// <summary>How a stretch of recorded speech was paced.</summary>
    public struct SpeechTiming
    {
        /// <summary>From the first sound of speech to the last, in seconds.</summary>
        public float SpanSeconds;

        /// <summary>Silences of a quarter of a second or more inside that span, added up.</summary>
        public float PauseSeconds;

        /// <summary>Anything above the noise floor was heard at all.</summary>
        public bool HasSpeech;
    }

    /// <summary>
    ///     A turn split into grammar, vocabulary and fluency, as percentages - what a teacher
    ///     asked for beside the single fluency score.
    ///
    ///     <para>
    ///     Two of the three are measured here rather than asked of the model, because the model's
    ///     own number moves when nothing has changed: the same recording has come back 45, 68
    ///     and 45. Grammar is how much of what was said the coach left alone, counted word by
    ///     word from its correction. Fluency is how fast the learner spoke and how much of it was
    ///     silence, measured from the recording itself. Only vocabulary - whether the words fit
    ///     the situation - needs judgement, and comes from the coach.
    ///     </para>
    ///
    ///     <para>
    ///     None of the three feeds the main score, which is still the coach's. They sit beside it.
    ///     Free of UnityEngine, like the rest of this assembly.
    ///     </para>
    /// </summary>
    public static class TurnSkills
    {
        /// <summary>Not measured for this turn: typed, too short, or no correction data.</summary>
        public const int NotMeasured = -1;

        /// <summary>
        ///     Words below which a single turn's percentages are not shown on their own: one slip
        ///     in "Can I have a latte?" is 80%, which says more about the length than the learner.
        ///     Shorter turns still count towards the averages.
        /// </summary>
        public const int MinimumWordsToShow = 8;

        /// <summary>Words in a turn, counted the way the rest of these rules count them.</summary>
        public static int CountWords(string text) => ProgressRules.Words(text).Count;

        // ──────────────────── grammar ────────────────────

        /// <summary>
        ///     The share of the turn the coach did not have to change, 0-100: one minus the
        ///     word-level edits in the correction over the length of what was said (or of the
        ///     corrected version, if longer - a single word made into a sentence is mostly
        ///     missing, not mostly right).
        ///
        ///     <para>
        ///     A correction that only swaps a word for a better one is vocabulary, not grammar,
        ///     and one that replaces the sentence with a different one is not a correction of it;
        ///     neither counts here. No correction at all is 100.
        ///     </para>
        /// </summary>
        public static int GrammarAccuracy(string utterance, string correction)
        {
            List<string> said = ProgressRules.Words(utterance);
            if (said.Count == 0) return NotMeasured;

            if (!TryReadPair(correction, out string original, out string corrected)) return 100;

            List<SkillArea> areas = SkillSummary.Classify(original, corrected);
            if (areas.Count == 0) return 100;
            if (areas.Count == 1 && areas[0] == SkillArea.WordChoice) return 100;

            List<string> o = ProgressRules.Words(original);
            List<string> c = ProgressRules.Words(corrected);
            int edits = EditDistance(o, c);
            int length = Math.Max(said.Count, c.Count);

            return Clamp((int)Math.Round(100.0 * (1.0 - (double)edits / length)));
        }

        private static bool TryReadPair(string correction, out string original, out string corrected)
        {
            original = corrected = "";
            if (string.IsNullOrWhiteSpace(correction)) return false;
            return SkillSummary.TryReadCorrection("• Grammar: " + correction.Trim(), out original, out corrected);
        }

        private static int EditDistance(List<string> a, List<string> b)
        {
            var d = new int[a.Count + 1, b.Count + 1];
            for (int i = 0; i <= a.Count; i++) d[i, 0] = i;
            for (int j = 0; j <= b.Count; j++) d[0, j] = j;
            for (int i = 1; i <= a.Count; i++)
                for (int j = 1; j <= b.Count; j++)
                    d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                                       d[i - 1, j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            return d[a.Count, b.Count];
        }

        // ──────────────────── fluency ────────────────────

        private const float FrameSeconds = 0.02f;
        private const float MinimumPauseSeconds = 0.25f;

        /// <summary>
        ///     Where the speech is in a recording, and how much silence it has inside it. A 20 ms
        ///     frame is speech when it is louder than both a fixed floor and an eighth of the
        ///     loudest frame, so a quiet microphone and a loud one are read alike.
        /// </summary>
        public static SpeechTiming MeasureTiming(float[] samples, int channels, int sampleRate)
        {
            var timing = new SpeechTiming();
            if (samples == null || samples.Length == 0 || sampleRate <= 0) return timing;
            channels = Math.Max(1, channels);

            int frame = Math.Max(1, (int)(sampleRate * FrameSeconds)) * channels;
            int frames = samples.Length / frame;
            if (frames == 0) return timing;

            var rms = new float[frames];
            float loudest = 0f;
            for (int f = 0; f < frames; f++)
            {
                double sum = 0;
                for (int i = f * frame; i < (f + 1) * frame; i++) sum += samples[i] * samples[i];
                rms[f] = (float)Math.Sqrt(sum / frame);
                if (rms[f] > loudest) loudest = rms[f];
            }

            float threshold = Math.Max(0.01f, loudest / 8f);
            int first = -1, last = -1;
            for (int f = 0; f < frames; f++)
            {
                if (rms[f] < threshold) continue;
                if (first < 0) first = f;
                last = f;
            }
            if (first < 0) return timing;

            int quiet = 0, pausedFrames = 0;
            int minimumPauseFrames = (int)Math.Ceiling(MinimumPauseSeconds / FrameSeconds);
            for (int f = first; f <= last; f++)
            {
                if (rms[f] < threshold) { quiet++; continue; }
                if (quiet >= minimumPauseFrames) pausedFrames += quiet;
                quiet = 0;
            }

            timing.HasSpeech = true;
            timing.SpanSeconds = (last - first + 1) * FrameSeconds;
            timing.PauseSeconds = pausedFrames * FrameSeconds;
            return timing;
        }

        /// <summary>Speech rate below which a turn scores nothing for pace.</summary>
        public const int SlowWordsPerMinute = 50;

        /// <summary>Speech rate from which a turn scores fully for pace - unhurried conversational English.</summary>
        public const int FluentWordsPerMinute = 130;

        /// <summary>
        ///     Fluency 0-100 from pace and pauses: six parts the speech rate (50 words a minute
        ///     scores nothing, 130 or more scores fully) and four parts how little of the time was
        ///     silence (a tenth or less scores fully, nearly half scores nothing). Not measured
        ///     for fewer than four words or a second and a half of speech, where one hesitation is
        ///     most of the turn.
        /// </summary>
        public static int Fluency(int words, SpeechTiming timing, out int wordsPerMinute)
        {
            wordsPerMinute = 0;
            if (!timing.HasSpeech || words < 4 || timing.SpanSeconds < 1.5f) return NotMeasured;

            wordsPerMinute = (int)Math.Round(words / timing.SpanSeconds * 60.0);
            double pace = (wordsPerMinute - SlowWordsPerMinute) / (double)(FluentWordsPerMinute - SlowWordsPerMinute);
            double pauseShare = timing.PauseSeconds / timing.SpanSeconds;
            double flow = (0.45 - pauseShare) / (0.45 - 0.10);

            return Clamp((int)Math.Round(100.0 * (0.6 * Unit(pace) + 0.4 * Unit(flow))));
        }

        // ──────────────────── vocabulary ────────────────────

        /// <summary>The coach's vocabulary score, or <see cref="NotMeasured"/> when it gave none or one out of range.</summary>
        public static int Vocabulary(int coachScore) =>
            coachScore >= 0 && coachScore <= 100 ? coachScore : NotMeasured;

        // ──────────────────── averages ────────────────────

        /// <summary>Adds a measured value to a running sum and count; leaves both alone for one not measured.</summary>
        public static void Add(int value, ref int sum, ref int count)
        {
            if (value < 0) return;
            sum += value;
            count++;
        }

        /// <summary>The rounded mean of a running sum, or 0 when nothing was measured.</summary>
        public static int Mean(int sum, int count) => count > 0 ? (int)Math.Round((double)sum / count) : 0;

        private static double Unit(double x) => x < 0 ? 0 : x > 1 ? 1 : x;

        private static int Clamp(int v) => v < 0 ? 0 : v > 100 ? 100 : v;
    }
}
