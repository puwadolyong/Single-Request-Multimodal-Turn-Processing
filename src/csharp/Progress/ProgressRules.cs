using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

namespace GeminiAvatar
{
    /// <summary>
    ///     The decisions <see cref="LearningProgressService"/> makes about a learner's record,
    ///     separated from the MonoBehaviour that makes them.
    ///
    ///     <para>
    ///     The service is a scene component wired to events, a clock and a save file, none of
    ///     which a test can supply cheaply, so the judgments inside it went unchecked: whether a
    ///     turn belongs in the review list, and how a practice streak advances. Both are pure
    ///     functions of their inputs once the current date is passed in rather than read, and
    ///     both have been wrong before in ways a learner would notice.
    ///     </para>
    ///
    ///     <para>
    ///     Deliberately free of UnityEngine, like the rest of this assembly, so it can be
    ///     exercised without the editor.
    ///     </para>
    /// </summary>
    public static class ProgressRules
    {
        /// <summary>Date format used for <see cref="LearningProgressData.LastPracticeDate"/>.</summary>
        public const string DateKeyFormat = "yyyy-MM-dd";

        /// <summary>
        ///     A turn at or above this score is treated as correct and not filed for review.
        ///     Shared by the progress record and the coach card, which offers the same sentence
        ///     to practise that the record keeps.
        /// </summary>
        public const int ReviewScoreThreshold = 90;

        /// <summary>
        ///     Whether a scored turn is worth putting in front of the learner again.
        ///
        ///     <para>
        ///     Expressed over the coach's findings rather than the score alone. The model scores
        ///     generously - it will award 92 and still rewrite the sentence - so a rule that only
        ///     looked at the number left the review list empty on turns where a real article or
        ///     tense error had just been corrected.
        ///     </para>
        /// </summary>
        /// <param name="scoreThreshold">Below this a turn is worth revisiting whatever else the coach said.</param>
        public static bool IsWorthReviewing(bool hasCorrection,
                                            bool hasPronunciationNote,
                                            string feedbackStatus,
                                            bool hasScore,
                                            int score,
                                            int scoreThreshold)
        {
            return hasCorrection
                   || hasPronunciationNote
                   || string.Equals(feedbackStatus, "NEEDS_FIX", StringComparison.OrdinalIgnoreCase)
                   || (hasScore && score < scoreThreshold);
        }

        /// <summary>
        ///     Records that the learner practised on <paramref name="today"/> and returns whether
        ///     anything changed.
        ///
        ///     <para>
        ///     The date is passed in rather than read from the clock so the behaviour at a day
        ///     boundary can be checked without waiting for one. Practising twice in a day counts
        ///     once; a gap of a single day continues the streak; any longer gap starts it over.
        ///     </para>
        /// </summary>
        public static bool AdvanceStreak(LearningProgressData data, DateTime today)
        {
            if (data == null)
                return false;

            string todayKey = today.Date.ToString(DateKeyFormat, CultureInfo.InvariantCulture);

            if (string.Equals(data.LastPracticeDate, todayKey, StringComparison.Ordinal))
                return false;

            bool hasPrevious = DateTime.TryParseExact(
                data.LastPracticeDate, DateKeyFormat,
                CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime last);

            // An unreadable stored date is treated as no previous day rather than as a gap,
            // so a corrupted field restarts the streak instead of throwing.
            if (hasPrevious && (today.Date - last.Date).Days == 1)
                data.CurrentStreakDays++;
            else
                data.CurrentStreakDays = 1;

            data.BestStreakDays = Math.Max(data.BestStreakDays, data.CurrentStreakDays);
            data.LastPracticeDate = todayKey;
            return true;
        }

        /// <summary>
        ///     Mean of a set of scores, rounded to the nearest whole number.
        ///     Empty or absent input averages to zero rather than throwing.
        /// </summary>
        public static int Average(IList<int> values)
        {
            if (values == null || values.Count == 0)
                return 0;

            long sum = 0;
            for (int i = 0; i < values.Count; i++)
                sum += values[i];

            // Midpoint-to-even, matching the UnityEngine.Mathf.RoundToInt this replaced, so
            // an average already stored in a save file does not shift by one on load.
            return (int)Math.Round((double)sum / values.Count);
        }

        /// <summary>
        ///     Caps a history at <paramref name="maxCount"/> entries by discarding the oldest,
        ///     which are at the front. A non-positive cap empties the list.
        /// </summary>
        public static void TrimFront<T>(List<T> list, int maxCount)
        {
            if (list == null)
                return;

            if (maxCount <= 0)
            {
                list.Clear();
                return;
            }

            int excess = list.Count - maxCount;
            if (excess > 0)
                list.RemoveRange(0, excess);
        }

        // ──────────────────── saying it again ────────────────────

        /// <summary>
        ///     Words fewer than which a corrected fragment is not worth saying on its own:
        ///     "I went" is the fix, not a sentence. The same four words
        ///     <c>ParsedResponse.MinimumObjectiveWords</c> uses, and measured the same way.
        /// </summary>
        public const int MinimumTargetWords = 4;

        /// <summary>
        ///     One word in this many may differ between the sentence and what was heard, so
        ///     long as it is not one of the words the correction changed. The recogniser
        ///     mishears - "sweet" came back as "switch" and "sandwich" in the same recording -
        ///     and a learner who said the sentence right should not be failed for that. Short
        ///     sentences get no slack at all.
        /// </summary>
        public const int WordsPerAllowedSlip = 6;

        /// <summary>
        ///     Words more than which a sentence is too long to say again from memory. A two-minute
        ///     IELTS long turn corrected in one place would otherwise come back whole as the
        ///     sentence to practise; past this, only the corrected phrase is offered.
        /// </summary>
        public const int MaximumTargetWords = 25;

        // Hyphenated words are one word, so "Wi-Fi" and "WiFi" compare equal once the hyphen
        // is dropped in Normalise.
        private static readonly Regex WordPattern =
            new Regex(@"[A-Za-z0-9’']+(?:-[A-Za-z0-9’']+)*", RegexOptions.Compiled);

        /// <summary>
        ///     Digits the recogniser may write for a number the target spells out, or the other
        ///     way round - "two nights" and "2 nights" are the same thing said. Values are the
        ///     words said, space separated: "21" and "twenty-one" both become "twenty one".
        /// </summary>
        private static readonly Dictionary<string, string> NumberWords = BuildNumberWords();

        private static Dictionary<string, string> BuildNumberWords()
        {
            string[] units =
            {
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen"
            };
            string[] tens = { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };

            var map = new Dictionary<string, string>();
            for (int n = 0; n < 20; n++)
                map[n.ToString(CultureInfo.InvariantCulture)] = units[n];
            for (int n = 20; n < 100; n++)
            {
                string spoken = n % 10 == 0 ? tens[n / 10] : tens[n / 10] + " " + units[n % 10];
                map[n.ToString(CultureInfo.InvariantCulture)] = spoken;
                // "twenty-one" loses its hyphen in Normalise and arrives as one word.
                if (n % 10 != 0) map[tens[n / 10] + units[n % 10]] = spoken;
            }
            map["100"] = "one hundred";
            return map;
        }

        /// <summary>
        ///     Contractions the recogniser writes for words a learner said in full, or the other
        ///     way round: "I'd like" for "I would like". The expanded word is usually the one the
        ///     correction added, which is never forgiven, so without this a sentence said right
        ///     failed on how it was written down. "'s" is left alone apart from a few pronouns -
        ///     after a noun it is as often a possessive as "is".
        /// </summary>
        private static readonly Dictionary<string, string> Contractions = new Dictionary<string, string>
        {
            ["i'm"] = "i am", ["won't"] = "will not", ["can't"] = "can not", ["cannot"] = "can not",
            ["shan't"] = "shall not", ["ain't"] = "ain't", ["it's"] = "it is", ["that's"] = "that is", ["what's"] = "what is",
            ["there's"] = "there is", ["here's"] = "here is", ["he's"] = "he is", ["she's"] = "she is",
            ["where's"] = "where is", ["who's"] = "who is", ["how's"] = "how is"
        };

        private static readonly string[][] ContractionEndings =
        {
            new[] { "n't", " not" }, new[] { "'re", " are" }, new[] { "'ve", " have" },
            new[] { "'ll", " will" }, new[] { "'d", " would" }
        };

        /// <summary>
        ///     A placeholder for the learner to fill in ("[Your Name]"), or a description of what
        ///     was said in place of what was said ("(No name provided)"). Neither can be read out.
        /// </summary>
        private static readonly Regex NotSayable = new Regex(@"[\[\]\(\)<>{}]", RegexOptions.Compiled);

        private static readonly Regex CorrectionShape = new Regex(
            @"Original:\s*(.*?)\s*->\s*Corrected:\s*(.*)$",
            RegexOptions.Singleline | RegexOptions.IgnoreCase | RegexOptions.Compiled);

        /// <summary>The sentence to say again, and the words in it the correction put there.</summary>
        public struct ReviewTargetResult
        {
            public string Sentence;

            /// <summary>
            ///     Words the corrected version has that the learner's did not - the point of
            ///     the exercise, and so never forgiven as a mishearing. Empty when the target
            ///     came from somewhere other than a correction.
            /// </summary>
            public List<string> KeyWords;

            public bool IsEmpty => string.IsNullOrWhiteSpace(Sentence);
        }

        /// <summary>
        ///     What the learner should say again: their own sentence with the coach's correction
        ///     applied, so they practise what they meant rather than a stranger's sentence.
        ///
        ///     <para>
        ///     The correction names a phrase, not a sentence ("I go -&gt; I went"), so it is put
        ///     back where it came from. When the phrase cannot be found in what was said - the
        ///     model paraphrased it - a corrected half long enough to stand alone is used; failing
        ///     that, the native phrasing; failing that, nothing, and no button is offered.
        ///     </para>
        /// </summary>
        public static ReviewTargetResult ReviewTarget(string userText, string correction, string betterPhrasing)
        {
            var result = new ReviewTargetResult { Sentence = "", KeyWords = new List<string>() };

            Match shape = CorrectionShape.Match((correction ?? "").Trim());
            if (shape.Success)
            {
                string original = Unquote(shape.Groups[1].Value);
                string corrected = Unquote(FirstAlternative(shape.Groups[2].Value));

                if (Words(corrected).Count > 0 && !NotSayable.IsMatch(corrected) && !NotSayable.IsMatch(original))
                {
                    // Either way it has to be a sentence: "A coffee please" is the fragment
                    // corrected, and saying a fragment again teaches nothing new.
                    string applied = ApplyCorrection(userText, original, corrected);
                    int appliedWords = Words(applied).Count;
                    string sentence = appliedWords >= MinimumTargetWords && appliedWords <= MaximumTargetWords
                        ? applied
                        : (Words(corrected).Count >= MinimumTargetWords && Words(corrected).Count <= MaximumTargetWords
                            ? Tidy(corrected) : "");

                    if (sentence.Length > 0 && !SameWords(sentence, userText))
                    {
                        result.Sentence = sentence;
                        result.KeyWords = WordsAdded(original, corrected);
                        return result;
                    }
                }
            }

            string better = Tidy(Unquote(FirstAlternative(betterPhrasing)));
            int betterWords = Words(better).Count;
            if (betterWords >= 2 && betterWords <= MaximumTargetWords && !NotSayable.IsMatch(better) && !SameWords(better, userText))
                result.Sentence = better;

            return result;
        }

        /// <summary>How a try at saying the target went.</summary>
        public struct RetryJudgement
        {
            public bool Passed;

            /// <summary>Nothing intelligible came back: not a failed try, just not a try.</summary>
            public bool HeardNothing;

            /// <summary>Word-level edits between the target and what was heard.</summary>
            public int WordDistance;

            /// <summary>Edits forgiven for a sentence this long.</summary>
            public int AllowedSlips;

            /// <summary>Target words that were not said, in order.</summary>
            public List<string> MissingWords;

            /// <summary>
            ///     Where <see cref="MissingWords"/> are in the target, as indexes into its words,
            ///     so the one actually missed is marked when a word appears twice.
            /// </summary>
            public List<int> MissingPositions;

            /// <summary>Words said that the target does not have, in order.</summary>
            public List<string> ExtraWords;

            /// <summary>Correction words that were not said. Any at all fails the try.</summary>
            public List<string> MissingKeyWords;
        }

        /// <summary>
        ///     Whether what was heard is the target sentence.
        ///
        ///     <para>
        ///     Decided here from the transcription rather than asked of the model. The model is
        ///     never told the target: told what to expect, a listener tends to hear it, and the
        ///     point is to find out what the learner actually said. Words are compared ignoring
        ///     case and punctuation, and a few slips are forgiven on a long sentence - but never
        ///     one of <paramref name="keyWords"/>, the words the correction was about.
        ///     </para>
        /// </summary>
        public static RetryJudgement JudgeRetry(string target, string heard, IList<string> keyWords = null)
        {
            List<string> t = Words(target);
            List<string> h = Words(heard);

            var judgement = new RetryJudgement
            {
                MissingWords = new List<string>(),
                MissingPositions = new List<int>(),
                ExtraWords = new List<string>(),
                MissingKeyWords = new List<string>(),
                AllowedSlips = t.Count / WordsPerAllowedSlip,
                HeardNothing = h.Count == 0
            };

            if (t.Count == 0 || h.Count == 0)
            {
                judgement.MissingWords.AddRange(t);
                for (int i = 0; i < t.Count; i++) judgement.MissingPositions.Add(i);
                judgement.WordDistance = t.Count;
                return judgement;
            }

            // Word-level edit distance, then walk it back to say which words differed.
            int[,] d = new int[t.Count + 1, h.Count + 1];
            for (int i = 0; i <= t.Count; i++) d[i, 0] = i;
            for (int j = 0; j <= h.Count; j++) d[0, j] = j;
            for (int i = 1; i <= t.Count; i++)
                for (int j = 1; j <= h.Count; j++)
                    d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                                       d[i - 1, j - 1] + (t[i - 1] == h[j - 1] ? 0 : 1));

            var matched = new bool[t.Count];
            int a = t.Count, b = h.Count;
            var missing = new List<string>();
            var extra = new List<string>();
            while (a > 0 || b > 0)
            {
                if (a > 0 && b > 0 && t[a - 1] == h[b - 1] && d[a, b] == d[a - 1, b - 1])
                {
                    matched[a - 1] = true;
                    a--; b--;
                }
                else if (a > 0 && b > 0 && d[a, b] == d[a - 1, b - 1] + 1)
                {
                    missing.Add(t[a - 1]);
                    extra.Add(h[b - 1]);
                    a--; b--;
                }
                else if (a > 0 && d[a, b] == d[a - 1, b] + 1)
                {
                    missing.Add(t[a - 1]);
                    a--;
                }
                else
                {
                    extra.Add(h[b - 1]);
                    b--;
                }
            }
            missing.Reverse();
            extra.Reverse();

            judgement.WordDistance = d[t.Count, h.Count];
            judgement.MissingWords = missing;
            judgement.ExtraWords = extra;

            // A word not matched by the alignment was either left out or said as something
            // else; either way it is the one missed.
            var remaining = new List<string>();
            for (int i = 0; i < t.Count; i++)
            {
                if (matched[i]) continue;
                remaining.Add(t[i]);
                judgement.MissingPositions.Add(i);
            }

            if (keyWords != null)
            {
                // A key word counts as said only where the alignment matched it, so "I like
                // would an iced latte" does not pass on having "would" somewhere.
                foreach (string raw in keyWords)
                    foreach (string k in Words(raw))
                        if (remaining.Contains(k) && !judgement.MissingKeyWords.Contains(k))
                            judgement.MissingKeyWords.Add(k);
            }

            judgement.Passed = judgement.WordDistance <= judgement.AllowedSlips
                               && judgement.MissingKeyWords.Count == 0;
            return judgement;
        }

        /// <summary>
        ///     Whether an entry should come up in today's review: it has something to say, has
        ///     not been mastered, and its next date has arrived.
        /// </summary>
        public static bool IsDueForReview(MistakeEntry entry, DateTime today)
        {
            if (entry == null || !entry.HasTarget || entry.IsMastered)
                return false;

            if (string.IsNullOrEmpty(entry.NextDueDateKey))
                return true;

            return !TryParseDateKey(entry.NextDueDateKey, out DateTime due) || today.Date >= due.Date;
        }

        /// <summary>
        ///     Records a try at an entry's target.
        ///
        ///     <para>
        ///     Mastered means said right on two different days. Twice in one sitting shows the
        ///     sentence can be repeated straight after hearing the fix, which is not the same as
        ///     having it the next day. Either way the entry comes back tomorrow until then.
        ///     </para>
        /// </summary>
        public static void RecordReviewResult(MistakeEntry entry, bool passed, DateTime now)
        {
            if (entry == null)
                return;

            string todayKey = now.Date.ToString(DateKeyFormat, CultureInfo.InvariantCulture);

            entry.ReviewAttempts++;
            entry.LastReviewIso = now.ToString("o", CultureInfo.InvariantCulture);
            entry.NextDueDateKey = now.Date.AddDays(1).ToString(DateKeyFormat, CultureInfo.InvariantCulture);

            if (!passed)
                return;

            entry.ReviewPasses++;

            if (!entry.IsMastered
                && !string.IsNullOrEmpty(entry.LastPassDateKey)
                && !string.Equals(entry.LastPassDateKey, todayKey, StringComparison.Ordinal))
            {
                entry.MasteredIso = now.ToString("o", CultureInfo.InvariantCulture);
            }

            entry.LastPassDateKey = todayKey;
        }

        /// <summary>
        ///     The newest unmastered entry that asks for the same sentence, so making the same
        ///     mistake again refreshes one entry instead of filling the list with copies.
        /// </summary>
        public static MistakeEntry FindOpenEntry(List<MistakeEntry> mistakes, string target)
        {
            if (mistakes == null || string.IsNullOrWhiteSpace(target))
                return null;

            for (int i = mistakes.Count - 1; i >= 0; i--)
            {
                MistakeEntry m = mistakes[i];
                if (m != null && m.HasTarget && !m.IsMastered && SameWords(m.Target, target))
                    return m;
            }
            return null;
        }

        /// <summary>The newest entry asking for this sentence, mastered or not.</summary>
        public static MistakeEntry FindEntry(List<MistakeEntry> mistakes, string target)
        {
            if (mistakes == null || string.IsNullOrWhiteSpace(target))
                return null;

            for (int i = mistakes.Count - 1; i >= 0; i--)
                if (mistakes[i] != null && mistakes[i].HasTarget && SameWords(mistakes[i].Target, target))
                    return mistakes[i];
            return null;
        }

        /// <summary>
        ///     Brings a save from before practising was possible up to date, and returns whether
        ///     it changed anything.
        ///
        ///     <para>
        ///     Version 1 entries have no sentence to practise, but the coach's notes they kept
        ///     carry what one is built from - the "Grammar:" line holds the correction and the
        ///     "Better Phrasing:" line the native phrasing - so the learner's existing review
        ///     list can be practised straight away instead of only from the next mistake on.
        ///     </para>
        /// </summary>
        public static bool UpgradeSave(LearningProgressData data)
        {
            if (data == null || data.Version >= LearningProgressData.CurrentVersion)
                return false;

            if (data.Mistakes != null)
            {
                foreach (MistakeEntry m in data.Mistakes)
                {
                    if (m == null || m.HasTarget) continue;

                    string correction = FeedbackLine(m.FeedbackText, "Grammar");
                    string better = FeedbackLine(m.FeedbackText, "Better Phrasing");
                    ReviewTargetResult t = ReviewTarget(m.UserText, correction, better);

                    m.Target = t.Sentence ?? "";
                    m.TargetKeyWords = JoinKeyWords(t.KeyWords);
                }
            }

            data.Version = LearningProgressData.CurrentVersion;
            return true;
        }

        /// <summary>The value of a "• Label: value" line in stored coach feedback.</summary>
        internal static string FeedbackLine(string feedback, string label)
        {
            if (string.IsNullOrEmpty(feedback)) return "";

            foreach (string raw in feedback.Replace("\r\n", "\n").Split('\n'))
            {
                string line = raw.Trim().TrimStart('•', '-', '*').Trim();
                if (line.StartsWith(label + ":", StringComparison.OrdinalIgnoreCase))
                    return line.Substring(label.Length + 1).Trim();
            }
            return "";
        }

        /// <summary>
        ///     Caps the review list, letting go of mastered sentences before ones still being
        ///     learned, oldest first in each case.
        /// </summary>
        /// <returns>The entries let go, oldest first.</returns>
        public static List<MistakeEntry> TrimMistakesKeepingOpen(List<MistakeEntry> mistakes, int maxCount)
        {
            var removed = new List<MistakeEntry>();
            if (mistakes == null)
                return removed;

            if (maxCount <= 0)
            {
                removed.AddRange(mistakes);
                mistakes.Clear();
                return removed;
            }

            while (mistakes.Count > maxCount)
            {
                int index = mistakes.FindIndex(m => m != null && m.IsMastered);
                if (index < 0) index = 0;
                removed.Add(mistakes[index]);
                mistakes.RemoveAt(index);
            }
            return removed;
        }

        /// <summary>
        ///     Caps the learner's review list as <see cref="TrimMistakesKeepingOpen(List{MistakeEntry}, int)"/>
        ///     does, keeping count of the practice sentences let go.
        ///
        ///     <para>
        ///     Mastered sentences are the first to go, so a count taken from the list alone fell
        ///     as the learner kept practising - the report a teacher reads went from 30 mastered
        ///     to none once the list filled up. The tallies keep what was let go in the totals.
        ///     </para>
        /// </summary>
        public static void TrimMistakesKeepingOpen(LearningProgressData data, int maxCount)
        {
            if (data == null)
                return;

            foreach (MistakeEntry m in TrimMistakesKeepingOpen(data.Mistakes, maxCount))
            {
                if (m == null || !m.HasTarget) continue;
                data.ArchivedSentences++;
                if (m.IsMastered) data.ArchivedSentencesMastered++;
            }
        }

        /// <summary>
        ///     <paramref name="text"/> with the words at <paramref name="positions"/> wrapped in the
        ///     given tags. Positions index the words as <see cref="JudgeRetry"/> counts them, so
        ///     a word that appears twice is marked where it was missed, not where it first
        ///     appears; a contraction is marked whole when either of its words was missed. The
        ///     rest of the text is left as it was.
        /// </summary>
        public static string MarkPositions(string text, IList<int> positions, string open, string close)
        {
            if (string.IsNullOrEmpty(text) || positions == null || positions.Count == 0)
                return text ?? "";

            // Which written word each counted word came from.
            var source = new List<int>();
            MatchCollection written = WordPattern.Matches(text);
            for (int i = 0; i < written.Count; i++)
                foreach (string _ in Expand(Normalise(written[i].Value)))
                    source.Add(i);

            var mark = new HashSet<int>();
            foreach (int p in positions)
                if (p >= 0 && p < source.Count) mark.Add(source[p]);

            var sb = new System.Text.StringBuilder(text.Length + mark.Count * 16);
            int last = 0;
            for (int i = 0; i < written.Count; i++)
            {
                if (!mark.Contains(i)) continue;
                Match m = written[i];
                sb.Append(text, last, m.Index - last).Append(open).Append(m.Value).Append(close);
                last = m.Index + m.Length;
            }
            sb.Append(text, last, text.Length - last);
            return sb.ToString();
        }

        /// <summary>Space-separated key words, the form they are stored in.</summary>
        public static string JoinKeyWords(IList<string> keyWords) =>
            keyWords == null ? "" : string.Join(" ", keyWords);

        /// <summary>Key words back from their stored form.</summary>
        public static List<string> SplitKeyWords(string stored) => Words(stored);

        // ── helpers ──

        /// <summary>
        ///     Lower-case words as said: contractions and numbers written out, so "I'd like 2"
        ///     and "I would like two" are the same five words.
        /// </summary>
        internal static List<string> Words(string text)
        {
            var words = new List<string>();
            if (string.IsNullOrEmpty(text)) return words;
            foreach (Match m in WordPattern.Matches(text))
                words.AddRange(Expand(Normalise(m.Value)));
            return words;
        }

        /// <summary>
        ///     Lower-case words as written, one per word in the text, for finding a phrase in
        ///     the learner's sentence where it has to be replaced in place.
        /// </summary>
        private static List<string> WrittenWords(string text)
        {
            var words = new List<string>();
            if (string.IsNullOrEmpty(text)) return words;
            foreach (Match m in WordPattern.Matches(text))
                words.Add(Normalise(m.Value));
            return words;
        }

        /// <summary>
        ///     One written word: lower case, hyphens dropped, and the quote marks around a word
        ///     in single quotes taken off - the apostrophe inside a contraction stays.
        /// </summary>
        private static string Normalise(string word) =>
            (word ?? "").Replace('’', '\'').Replace("-", "").Trim().Trim('\'').ToLowerInvariant();

        /// <summary>The words a written word stands for: itself, or a contraction or number written out.</summary>
        private static string[] Expand(string word)
        {
            if (word.Length == 0) return new string[0];
            if (NumberWords.TryGetValue(word, out string spelled)) return spelled.Split(' ');
            if (Contractions.TryGetValue(word, out string full)) return full.Split(' ');

            foreach (string[] ending in ContractionEndings)
                if (word.Length > ending[0].Length && word.EndsWith(ending[0], StringComparison.Ordinal))
                    return (word.Substring(0, word.Length - ending[0].Length) + ending[1]).Split(' ');

            return new[] { word };
        }

        /// <summary>
        ///     The first of "A / B" alternatives. The model sometimes offers two corrections,
        ///     and the learner can only be asked to say one.
        /// </summary>
        private static string FirstAlternative(string text)
        {
            string s = text ?? "";
            int slash = s.IndexOf(" / ", StringComparison.Ordinal);
            return slash > 0 ? s.Substring(0, slash) : s;
        }

        private static bool SameWords(string a, string b)
        {
            List<string> x = Words(a), y = Words(b);
            if (x.Count != y.Count) return false;
            for (int i = 0; i < x.Count; i++)
                if (x[i] != y[i]) return false;
            return true;
        }

        /// <summary>Words of the corrected half the original half did not have.</summary>
        private static List<string> WordsAdded(string original, string corrected)
        {
            var left = Words(original);
            var added = new List<string>();
            foreach (string w in Words(corrected))
            {
                int at = left.IndexOf(w);
                if (at >= 0) left.RemoveAt(at);
                else if (!added.Contains(w)) added.Add(w);
            }
            return added;
        }

        /// <summary>
        ///     The learner's sentence with <paramref name="original"/> replaced by
        ///     <paramref name="corrected"/>, matched word by word so that case and punctuation do
        ///     not hide it. Empty when the phrase is not in the sentence.
        /// </summary>
        private static string ApplyCorrection(string userText, string original, string corrected)
        {
            if (string.IsNullOrWhiteSpace(userText)) return "";

            MatchCollection said = WordPattern.Matches(userText);
            List<string> find = WrittenWords(original);
            if (find.Count == 0 || said.Count < find.Count) return "";

            for (int start = 0; start + find.Count <= said.Count; start++)
            {
                bool hit = true;
                for (int k = 0; k < find.Count && hit; k++)
                    hit = Normalise(said[start + k].Value) == find[k];
                if (!hit) continue;

                int from = said[start].Index;
                Match last = said[start + find.Count - 1];
                int to = last.Index + last.Length;

                // At the end of what was said, the corrected phrase brings its own full stop or
                // question mark - "Total price" has none to keep. Inside the sentence, the
                // learner's own punctuation around it stays.
                string rest = userText.Substring(to);
                bool atEnd = rest.Trim().TrimEnd('.', '!', '?', ',').Length == 0;
                string fixedPhrase = corrected.Trim();
                bool bringsItsOwnEnd = fixedPhrase.EndsWith(".") || fixedPhrase.EndsWith("?") || fixedPhrase.EndsWith("!");
                string rebuilt = atEnd
                    ? userText.Substring(0, from) + (bringsItsOwnEnd ? fixedPhrase : fixedPhrase + rest.Trim())
                    : userText.Substring(0, from) + corrected.TrimEnd('.', '!', '?', ',', ' ') + rest;
                return Tidy(rebuilt);
            }

            return "";
        }

        private static string Unquote(string text)
        {
            string s = (text ?? "").Trim();
            const string quotes = "\"'“”‘’`";
            while (s.Length >= 2 && quotes.IndexOf(s[0]) >= 0 && quotes.IndexOf(s[s.Length - 1]) >= 0)
                s = s.Substring(1, s.Length - 2).Trim();
            return s;
        }

        /// <summary>Single spaces, no space before punctuation, capital first letter.</summary>
        private static string Tidy(string text)
        {
            string s = Regex.Replace(text ?? "", @"\s+", " ").Trim();
            s = Regex.Replace(s, @"\s+([.,!?;:])", "$1");
            s = Regex.Replace(s, @"([.!?])\1+", "$1");
            if (s.Length > 0 && char.IsLower(s[0]))
                s = char.ToUpperInvariant(s[0]) + s.Substring(1);
            return s;
        }

        private static bool TryParseDateKey(string key, out DateTime date) =>
            DateTime.TryParseExact(key, DateKeyFormat, CultureInfo.InvariantCulture, DateTimeStyles.None, out date);
    }
}
