using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;

namespace GeminiAvatar
{
    /// <summary>A kind of mistake the coach corrects, named the way a teacher would group them.</summary>
    public enum SkillArea
    {
        /// <summary>A word or two said where a sentence was needed: "Card" for "I'd like to pay by card".</summary>
        FullSentences,
        Articles,
        VerbForms,
        Plurals,
        /// <summary>to, is, am, are, do: the small words Thai has no need for.</summary>
        LinkingWords,
        Prepositions,
        WordOrder,
        PoliteRequests,
        Pronunciation,
        /// <summary>Anything else the coach changed: usually a better word.</summary>
        WordChoice
    }

    /// <summary>One line of the overview, in English and Thai.</summary>
    public struct SummaryPoint
    {
        public string En;
        public string Th;

        /// <summary>A sentence the learner was given to say, as an example; may be empty.</summary>
        public string Example;
    }

    /// <summary>How often one kind of mistake came up.</summary>
    public struct AreaCount
    {
        public SkillArea Area;
        public int Count;
        public string Example;
    }

    /// <summary>
    ///     What a learner is doing well and what to practise next, from their record - the
    ///     overview a teacher asked for at the top of the report.
    ///
    ///     <para>
    ///     The mistakes are grouped by comparing what was said with the coach's correction
    ///     word by word, not by asking the model: words added or dropped, a word changed into a
    ///     form of itself, the same words in a new order. It is a teacher's rough sort rather
    ///     than linguistics - "I want order" missing "to" is counted as a linking word, not as a
    ///     verb pattern - but it is the same on every report, costs nothing, and can be checked.
    ///     The groups were chosen from 815 corrections the coach actually made.
    ///     </para>
    ///
    ///     <para>
    ///     Free of UnityEngine, like the rest of this assembly.
    ///     </para>
    /// </summary>
    public static class SkillSummary
    {
        private static readonly HashSet<string> ArticleWords = new HashSet<string> { "a", "an", "the" };
        private static readonly HashSet<string> PoliteWords = new HashSet<string> { "would", "could", "please", "may" };
        private static readonly HashSet<string> LinkWords = new HashSet<string>
            { "to", "is", "am", "are", "was", "were", "be", "do", "does", "did", "have", "has" };
        private static readonly HashSet<string> PrepositionWords = new HashSet<string>
            { "at", "in", "on", "for", "of", "from", "with", "by", "about", "into", "under" };

        /// <summary>Irregular past and participle forms the learners in the logs got wrong.</summary>
        private static readonly Dictionary<string, string[]> Irregular = new Dictionary<string, string[]>
        {
            ["go"] = new[] { "went", "gone", "going" }, ["come"] = new[] { "came", "coming" },
            ["take"] = new[] { "took", "taken" }, ["buy"] = new[] { "bought" }, ["have"] = new[] { "had" },
            ["eat"] = new[] { "ate", "eaten" }, ["get"] = new[] { "got" }, ["make"] = new[] { "made" },
            ["see"] = new[] { "saw", "seen" }, ["give"] = new[] { "gave", "given" }, ["pay"] = new[] { "paid" },
            ["leave"] = new[] { "left" }, ["bring"] = new[] { "brought" }, ["stay"] = new[] { "stayed" },
            ["tell"] = new[] { "told" }, ["say"] = new[] { "said" }, ["find"] = new[] { "found" },
            ["lose"] = new[] { "lost" }, ["send"] = new[] { "sent" }, ["think"] = new[] { "thought" },
        };

        private static readonly Regex CorrectionShape = new Regex(
            @"Original:\s*(.*?)\s*->\s*Corrected:\s*(.*)$", RegexOptions.Singleline | RegexOptions.IgnoreCase);

        // ──────────────────── grouping one correction ────────────────────

        /// <summary>
        ///     The kinds of mistake one correction fixed. Empty when nothing was changed - the
        ///     coach sometimes "corrects" a sentence into itself.
        /// </summary>
        public static List<SkillArea> Classify(string original, string corrected)
        {
            var areas = new List<SkillArea>();

            // "(Asking for the total before ordering)" describes what was said rather than
            // quoting it; there is nothing in it to have got wrong.
            if (Unsayable.IsMatch(original ?? "")) return areas;

            List<string> o = WithoutRepeat(ProgressRules.Words(original));
            List<string> c = ProgressRules.Words(FirstAlternative(corrected));
            if (o.Count == 0 || c.Count == 0 || SameSequence(o, c) || IsRepeatOf(o, c)) return areas;

            // A fragment made into a sentence. Nothing else about it is worth counting: every
            // word the sentence needed is "added".
            if (o.Count <= 3 && c.Count >= o.Count + 2)
            {
                areas.Add(SkillArea.FullSentences);
                return areas;
            }

            List<string> added = Minus(c, o), removed = Minus(o, c);

            // A different sentence altogether - the coach suggesting what to say next rather
            // than repairing what was said - is not a mistake in this one.
            if (o.Count - removed.Count < (o.Count + 1) / 2) return areas;
            var changed = new List<string>(added);
            changed.AddRange(removed);

            if (added.Count == 0 && removed.Count == 0)
            {
                areas.Add(SkillArea.WordOrder);
                return areas;
            }

            if (changed.Exists(ArticleWords.Contains)) areas.Add(SkillArea.Articles);
            if (added.Exists(PoliteWords.Contains)) areas.Add(SkillArea.PoliteRequests);
            if (changed.Exists(LinkWords.Contains)) areas.Add(SkillArea.LinkingWords);
            if (changed.Exists(PrepositionWords.Contains)) areas.Add(SkillArea.Prepositions);

            bool plural = false, form = false;
            foreach (string a in added)
            {
                foreach (string r in removed)
                {
                    if (IsPluralOf(a, r) || IsPluralOf(r, a)) plural = true;
                    else if (IsFormOf(a, r)) form = true;
                }
            }
            if (plural) areas.Add(SkillArea.Plurals);
            if (form) areas.Add(SkillArea.VerbForms);

            if (areas.Count == 0) areas.Add(SkillArea.WordChoice);
            return areas;
        }

        /// <summary>The original and corrected halves of a stored "Grammar:" line, if it has one.</summary>
        public static bool TryReadCorrection(string feedbackText, out string original, out string corrected)
        {
            original = corrected = "";
            string line = ProgressRules.FeedbackLine(feedbackText, "Grammar");
            Match m = CorrectionShape.Match(line);
            if (!m.Success) return false;
            original = m.Groups[1].Value.Trim().Trim('"', '\'', '“', '”');
            corrected = m.Groups[2].Value.Trim().Trim('"', '\'', '“', '”');
            return original.Length > 0 && corrected.Length > 0;
        }

        /// <summary>
        ///     How many corrections in the record fall in each area, most first. Pronunciation
        ///     comes from the coach's pronunciation notes rather than from a correction.
        /// </summary>
        public static List<AreaCount> CountAreas(LearningProgressData data)
        {
            var counts = new Dictionary<SkillArea, AreaCount>();
            var pureExample = new HashSet<SkillArea>();
            if (data?.Mistakes == null) return new List<AreaCount>();

            foreach (MistakeEntry m in data.Mistakes)
            {
                if (m == null) continue;
                string example = (m.Target ?? "").Trim();

                if (TryReadCorrection(m.FeedbackText, out string original, out string corrected))
                {
                    List<SkillArea> areas = Classify(original, corrected);
                    foreach (SkillArea area in areas)
                        Tally(counts, pureExample, area, example.Length > 0 ? example : corrected, areas.Count == 1);
                }

                if (ProgressRules.FeedbackLine(m.FeedbackText, "Pronunciation").Length > 0)
                    Tally(counts, pureExample, SkillArea.Pronunciation, "", false);
            }

            var list = new List<AreaCount>(counts.Values);
            list.Sort((x, y) => y.Count != x.Count ? y.Count.CompareTo(x.Count) : x.Area.CompareTo(y.Area));
            return list;
        }

        /// <param name="pure">
        ///     The correction fixed only this kind of mistake. Such an example is kept over one
        ///     that also changed other things, so the sentence shown shows the point.
        /// </param>
        private static void Tally(Dictionary<SkillArea, AreaCount> counts, HashSet<SkillArea> pureExample,
                                  SkillArea area, string example, bool pure)
        {
            counts.TryGetValue(area, out AreaCount c);
            c.Area = area;
            c.Count++;
            if (!string.IsNullOrEmpty(example) && !pureExample.Contains(area) && (pure || string.IsNullOrEmpty(c.Example)))
            {
                c.Example = example;
                if (pure) pureExample.Add(area);
            }
            counts[area] = c;
        }

        // ──────────────────── the overview ────────────────────

        /// <summary>A scenario at or above this average is something the learner does well.</summary>
        public const int StrongAverage = 85;

        /// <summary>A scenario below this average is one to go back to.</summary>
        public const int WeakAverage = 75;

        /// <summary>
        ///     What the learner does well. Each point is a fact from the record, so an empty list
        ///     means there is not yet enough practice to say anything, never that nothing went well.
        /// </summary>
        public static List<SummaryPoint> Strengths(LearningProgressData data)
        {
            var points = new List<SummaryPoint>();
            if (data == null) return points;

            (int first, int last)? trend = Trend(data);
            if (trend.HasValue && trend.Value.last - trend.Value.first >= 5)
                points.Add(Point($"Fluency has improved, from {trend.Value.first} to {trend.Value.last}",
                                 $"คะแนนดีขึ้น จาก {trend.Value.first} เป็น {trend.Value.last}"));

            var strong = new List<string>();
            foreach (ScenarioStat s in data.Scenarios)
                if (s != null && s.BestAverageScore >= StrongAverage)
                    strong.Add(Friendly(s.ScenarioName) + " (" + s.BestAverageScore + ")");
            if (strong.Count > 0)
                points.Add(Point("Speaks confidently in " + string.Join(", ", strong),
                                 "ทำได้ดีในสถานการณ์ " + string.Join(", ", strong)));

            if (data.ScenariosCleared > 0)
                points.Add(Point($"Completed every objective in {Plural(data.ScenariosCleared, "scenario")}",
                                 $"ทำภารกิจครบใน {data.ScenariosCleared} สถานการณ์"));

            int days = PracticeDays(data);
            if (days >= 3)
                points.Add(Point($"Practises regularly: on {days} different days, best run {Plural(data.BestStreakDays, "day")} in a row",
                                 $"ฝึกสม่ำเสมอ {days} วัน ต่อเนื่องสูงสุด {data.BestStreakDays} วัน"));

            if (data.SentencesMastered > 0)
                points.Add(Point($"Has mastered {Plural(data.SentencesMastered, "corrected sentence")}, said right on two different days",
                                 $"จำประโยคที่เคยผิดได้แล้ว {data.SentencesMastered} ประโยค"));

            return points;
        }

        /// <summary>
        ///     What to practise next, most useful first: the mistakes that keep coming back,
        ///     then the situation that went least well, then habits - corrections not practised,
        ///     a long gap, scores going down.
        /// </summary>
        /// <param name="asOf">The report's date, for how long ago the last practice was.</param>
        public static List<SummaryPoint> FocusAreas(LearningProgressData data, DateTime asOf, int maxAreas = 3)
        {
            var points = new List<SummaryPoint>();
            if (data == null) return points;

            int shown = 0;
            foreach (AreaCount a in CountAreas(data))
            {
                // Once is a slip; twice is a pattern. "Word choice" is the remainder, not a
                // pattern, so it is only named when it is all there is.
                if (a.Count < 2 || shown >= maxAreas) continue;
                if (a.Area == SkillArea.WordChoice && shown > 0) continue;

                (string en, string th) = Describe(a.Area);
                points.Add(new SummaryPoint
                {
                    En = en + " — " + Plural(a.Count, "correction"),
                    Th = th + " · " + a.Count + " ครั้ง",
                    Example = a.Area == SkillArea.Pronunciation ? "" : a.Example
                });
                shown++;
            }

            ScenarioStat weakest = null;
            foreach (ScenarioStat s in data.Scenarios)
                if (s != null && s.LastAverageScore > 0 && s.LastAverageScore < WeakAverage &&
                    (weakest == null || s.LastAverageScore < weakest.LastAverageScore))
                    weakest = s;
            if (weakest != null)
                points.Add(Point($"{Friendly(weakest.ScenarioName)} needs another go (latest average {weakest.LastAverageScore})",
                                 $"ควรฝึกสถานการณ์ {Friendly(weakest.ScenarioName)} อีกครั้ง (คะแนนล่าสุด {weakest.LastAverageScore})"));

            int unpractised = 0;
            foreach (MistakeEntry m in data.Mistakes)
                if (m != null && m.HasTarget && !m.IsMastered && m.ReviewAttempts == 0) unpractised++;
            if (unpractised >= 3)
                points.Add(Point($"{Plural(unpractised, "corrected sentence")} not practised yet — use \"Say it correctly\" or \"Review today\"",
                                 $"มีประโยคที่ยังไม่ได้ฝึกซ้ำ {unpractised} ประโยค ใช้ปุ่ม \"ลองพูดให้ถูก\" หรือ \"ทบทวนวันนี้\""));

            if (DateTime.TryParseExact(data.LastPracticeDate, ProgressRules.DateKeyFormat, CultureInfo.InvariantCulture,
                                       DateTimeStyles.None, out DateTime last))
            {
                int gap = (asOf.Date - last.Date).Days;
                if (gap >= 7)
                    points.Add(Point($"No practice in the last {gap} days — a few minutes a day works better than a long session",
                                     $"ไม่ได้ฝึกมา {gap} วัน ฝึกวันละไม่กี่นาทีได้ผลดีกว่าฝึกนานๆ ครั้ง"));
            }

            (int first, int last)? trend = Trend(data);
            if (trend.HasValue && trend.Value.first - trend.Value.last >= 5)
                points.Add(Point($"Recent scores are lower than at the start ({trend.Value.first} → {trend.Value.last})",
                                 $"คะแนนช่วงหลังต่ำกว่าช่วงแรก ({trend.Value.first} → {trend.Value.last})"));

            return points;
        }

        /// <summary>The name of an area and what it covers, in English and Thai.</summary>
        public static (string En, string Th) Describe(SkillArea area)
        {
            switch (area)
            {
                case SkillArea.FullSentences: return ("Answering in full sentences, not single words", "ตอบเป็นประโยคเต็ม ไม่ใช่คำเดียว");
                case SkillArea.Articles: return ("Articles: a, an, the", "คำนำหน้านาม a, an, the");
                case SkillArea.VerbForms: return ("Word forms and tenses (go → went, careful → carefully)", "รูปคำและกาล เช่น go → went");
                case SkillArea.Plurals: return ("Plurals (two days, five dollars)", "คำนามพหูพจน์ เติม -s");
                case SkillArea.LinkingWords: return ("Small words: to, is, am, are, do", "คำเล็กๆ ที่ขาดไม่ได้ เช่น to, is, am, do");
                case SkillArea.Prepositions: return ("Prepositions: at, in, on, for", "คำบุพบท at, in, on, for");
                case SkillArea.WordOrder: return ("Word order, especially in questions", "การเรียงคำ โดยเฉพาะในประโยคคำถาม");
                case SkillArea.PoliteRequests: return ("Polite requests: Could I…, I would like…", "การขออย่างสุภาพ Could I…, I would like…");
                case SkillArea.Pronunciation: return ("Pronunciation", "การออกเสียง");
                default: return ("Choosing the right word", "การเลือกใช้คำ");
            }
        }

        // ── helpers ──

        private static SummaryPoint Point(string en, string th) => new SummaryPoint { En = en, Th = th, Example = "" };

        private static string Plural(int n, string noun) => n + " " + (n == 1 ? noun : noun + "s");

        /// <summary>First and latest scored session averages, oldest first.</summary>
        private static (int first, int last)? Trend(LearningProgressData data)
        {
            var scored = new List<SessionRecord>();
            foreach (SessionRecord s in data.Sessions)
                if (s != null && s.AverageScore > 0) scored.Add(s);
            if (scored.Count < 2) return null;
            return (scored[0].AverageScore, scored[scored.Count - 1].AverageScore);
        }

        private static int PracticeDays(LearningProgressData data)
        {
            var days = new HashSet<string>();
            foreach (SessionRecord s in data.Sessions)
                if (s != null && DateTime.TryParse(s.StartedAtIso, CultureInfo.InvariantCulture,
                                                   DateTimeStyles.RoundtripKind, out DateTime d))
                    days.Add(d.ToString(ProgressRules.DateKeyFormat, CultureInfo.InvariantCulture));
            return days.Count;
        }

        /// <summary>"CafeRestaurant" → "Cafe Restaurant", or the edition's own name, the way the report names scenarios.</summary>
        private static string Friendly(string name) => ReportLocale.FriendlyScenario(name);

        /// <summary>
        ///     The words once, when a client bug sent them twice in a row. Corrections recorded
        ///     then read "Original: X.X -> Corrected: X".
        /// </summary>
        private static List<string> WithoutRepeat(List<string> words)
        {
            int n = words.Count;
            if (n >= 2 && n % 2 == 0)
            {
                bool twice = true;
                for (int i = 0; i < n / 2 && twice; i++) twice = words[i] == words[i + n / 2];
                if (twice) return words.GetRange(0, n / 2);
            }
            return words;
        }

        private static readonly Regex Unsayable = new Regex(@"[\[\]\(\)<>{}]");

        /// <summary>
        ///     What was said is the correction followed by the start of it again - the same
        ///     double-send, cut short with "...". The correction changed nothing.
        /// </summary>
        private static bool IsRepeatOf(List<string> said, List<string> corrected)
        {
            if (said.Count <= corrected.Count) return false;
            for (int i = 0; i < corrected.Count; i++)
                if (said[i] != corrected[i]) return false;
            for (int i = corrected.Count; i < said.Count; i++)
                if (i - corrected.Count >= corrected.Count || said[i] != corrected[i - corrected.Count]) return false;
            return true;
        }

        private static string FirstAlternative(string text)
        {
            string s = text ?? "";
            int slash = s.IndexOf(" / ", StringComparison.Ordinal);
            return slash > 0 ? s.Substring(0, slash) : s;
        }

        private static bool SameSequence(List<string> a, List<string> b)
        {
            if (a.Count != b.Count) return false;
            for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
            return true;
        }

        /// <summary>The words of <paramref name="a"/> left after taking out one of each in <paramref name="b"/>.</summary>
        private static List<string> Minus(List<string> a, List<string> b)
        {
            var rest = new List<string>(b);
            var result = new List<string>();
            foreach (string w in a)
            {
                int at = rest.IndexOf(w);
                if (at >= 0) rest.RemoveAt(at);
                else result.Add(w);
            }
            return result;
        }

        private static bool IsPluralOf(string plural, string single) =>
            plural == single + "s" || plural == single + "es" ||
            (single.EndsWith("y") && plural == single.Substring(0, single.Length - 1) + "ies");

        /// <summary>One word a form of the other: a shared stem of four letters, or an irregular pair.</summary>
        private static bool IsFormOf(string a, string b)
        {
            if (a.Length >= 4 && b.Length >= 4 && string.CompareOrdinal(a, 0, b, 0, 4) == 0) return true;
            if (a.Length >= 3 && b.Length >= 3 && Math.Min(a.Length, b.Length) <= 4 &&
                string.CompareOrdinal(a, 0, b, 0, 3) == 0) return true;
            return Irregular.TryGetValue(a, out string[] fa) && Array.IndexOf(fa, b) >= 0
                || Irregular.TryGetValue(b, out string[] fb) && Array.IndexOf(fb, a) >= 0;
        }
    }
}
