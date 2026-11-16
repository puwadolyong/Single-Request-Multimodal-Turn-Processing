using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using Newtonsoft.Json;

namespace GeminiAvatar
{
    /// <summary>
    ///     One coaching turn, normalised into the shape the avatar, the UI and the progress
    ///     tracker consume.
    ///
    ///     <para>
    ///     <see cref="Parse"/> accepts two wire formats. The first choice is the structured
    ///     JSON object described by <see cref="CoachTurnSchema"/>, which the API is asked to
    ///     produce via <c>responseSchema</c>. If the model returns prose instead — an older
    ///     model, a refusal, a schema the API declined — parsing falls back to the original
    ///     <c>[FEEDBACK]</c>/<c>[SPOKEN]</c> tag format so a deployed build keeps working.
    ///     </para>
    ///
    ///     <para>
    ///     This type deliberately has no UnityEngine dependency: it lives in its own assembly
    ///     so the parsing rules can be unit-tested without entering play mode.
    ///     </para>
    /// </summary>
    public class ParsedResponse
    {
        /// <summary>Clean dialogue text to speak. Empty when the model produced nothing usable.</summary>
        public string Text = "";

        /// <summary>The learner's speech, transcribed. Empty for typed input.</summary>
        public string UserTranscription = "";

        /// <summary>Coach feedback, formatted for display.</summary>
        public string FeedbackText = "";

        /// <summary>
        ///     The highest objective number any scenario has: the job interview's ten. An id
        ///     above it is not an objective at all; the controller then holds each scenario to
        ///     its own count.
        /// </summary>
        public const int MaxObjectiveId = 10;

        /// <summary>PERFECT, GOOD or NEEDS_FIX.</summary>
        public string FeedbackStatus = "GOOD";

        /// <summary>
        ///     The corrected sentence, when the coach rewrote something. Empty when the line was
        ///     already correct.
        /// </summary>
        public string Correction = "";

        /// <summary>
        ///     True when the coach actually rewrote the learner's sentence.
        ///
        ///     <para>
        ///     This, not the score, is the honest signal that a line is worth revisiting: the
        ///     model scores generously and will happily give 92 while still correcting a missing
        ///     article.
        ///     </para>
        /// </summary>
        public bool HasCorrection;

        /// <summary>Pronunciation note taken from the audio. Empty when typed or when nothing stood out.</summary>
        public string PronunciationNote = "";

        /// <summary>True when the coach had something to say about pronunciation.</summary>
        public bool HasPronunciationNote;

        /// <summary>The coach's short grammar tip, in English.</summary>
        public string Tip = "";

        /// <summary>The grammar tip restated in Thai, when Thai explanations are on.</summary>
        public string TipThai = "";

        /// <summary>
        ///     The natural phrasing a native speaker would use, on its own.
        ///
        ///     <para>
        ///     Also folded into <see cref="FeedbackText"/> for display, but kept separately
        ///     because the display copy is rich text mixed with every other section, and the
        ///     learner wants to hear this sentence spoken - which needs the sentence, not a
        ///     paragraph with colour tags in it.
        ///     </para>
        /// </summary>
        public string BetterPhrasing = "";

        /// <summary>
        ///     Fluency score 0-100. Meaningful only when <see cref="HasScore"/> is true —
        ///     an unscored turn reports 0 rather than a made-up number, so it can be
        ///     excluded from averages instead of skewing them.
        /// </summary>
        public int Score;

        /// <summary>True when the model actually returned a score for this turn.</summary>
        public bool HasScore;

        /// <summary>
        ///     Things the learner could say next, at most three, for the "Need an idea?" button.
        ///     Empty when the coach gave none.
        /// </summary>
        public List<Suggestion> Suggestions = new List<Suggestion>();

        /// <summary>The avatar's reply in Thai, for a beginner; empty otherwise.</summary>
        public string ReplyThai = "";

        /// <summary>The coach's word-choice score, 0-100, or -1 when it gave none.</summary>
        public int VocabularyScore = -1;

        /// <summary>
        ///     Share of the turn the coach left alone, 0-100, or -1. Worked out on the client from
        ///     the correction (TurnSkills.GrammarAccuracy), not returned by the model.
        /// </summary>
        public int GrammarScore = -1;

        /// <summary>
        ///     Pace and pauses, 0-100, or -1 for a typed or very short turn. Measured on the client
        ///     from the recording (TurnSkills.Fluency), not returned by the model.
        /// </summary>
        public int FluencyScore = -1;

        /// <summary>Speech rate of the recording, or 0 when not measured.</summary>
        public int WordsPerMinute;

        /// <summary>Seconds from the first sound of speech to the last, or -1 for a typed turn. From the recording.</summary>
        public float SpeechSeconds = -1f;

        /// <summary>Pauses of a quarter second or more inside the speech, in seconds, or -1 for a typed turn.</summary>
        public float PauseSeconds = -1f;

        /// <summary>Whether there is feedback worth showing.</summary>
        public bool HasFeedback;

        /// <summary>Emotion for the avatar to play.</summary>
        public AvatarEmotion Emotion = AvatarEmotion.Neutral;

        /// <summary>Emotion strength, 1-3.</summary>
        public int EmotionIntensity = 2;

        /// <summary>Quest objectives (1-3) accomplished this turn.</summary>
        public List<int> CompletedQuests = new List<int>();

        /// <summary>
        ///     Lowest score the rubric calls "nothing at all to correct". A turn the coach
        ///     rewrote cannot sit at or above this.
        /// </summary>
        public const int FlawlessBandFloor = 90;

        /// <summary>
        ///     Lowest score that still counts as having said a sentence. Below this the rubric
        ///     calls the turn "not a sentence, or barely intelligible".
        /// </summary>
        public const int SentenceBandFloor = 50;

        /// <summary>
        ///     Fewest words an utterance may have and still clear an objective.
        ///
        ///     <para>
        ///     Measured rather than chosen: over 512 recorded live turns the shortest utterance
        ///     that was a sentence ran to five words, and the longest bare fragment to three, so
        ///     four separates them with a word of margin on each side.
        ///     </para>
        /// </summary>
        public const int MinimumObjectiveWords = 4;

        /// <summary>
        ///     Highest score a fragment may have. The rubric's own ceiling rule: "if the learner
        ///     did not produce a complete sentence ... the score cannot exceed 55".
        /// </summary>
        public const int FragmentScoreCeiling = 55;

        /// <summary>
        ///     Longest utterance the fragment ceiling looks at.
        ///
        ///     <para>
        ///     Two, because that is where a model was seen breaking the rule it was given:
        ///     gemini-3.5-flash-lite scored "Painkiller" 100 and "No allergy" 88, and at 90 the
        ///     application plays a cheer - for one word. Longer than two and a short sentence
        ///     ("Yes, I do.") is too easily caught, which would be worse than the fault.
        ///     </para>
        /// </summary>
        public const int FragmentMaxWords = 2;

        /// <summary>
        ///     Verbs that make a one- or two-word utterance a sentence rather than a fragment -
        ///     "Sit down.", "Come in.", "Thank you." - in the imperative, where there is no
        ///     subject to count.
        ///
        ///     <para>
        ///     This list only ever exempts. A verb that is missing from it means the model is
        ///     overruled on a real sentence; a word wrongly on it means the model's own score
        ///     stands, which is what happened before this rule existed. So it is kept short and
        ///     errs towards trusting the model.
        ///     </para>
        /// </summary>
        private static readonly HashSet<string> ImperativeVerbs = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "be", "bring", "call", "close", "come", "drink", "eat", "enjoy", "excuse", "follow",
            "give", "go", "help", "hold", "hurry", "keep", "let", "listen", "look", "move",
            "open", "read", "relax", "repeat", "see", "show", "sit", "sleep", "speak", "stand",
            "start", "stop", "take", "tell", "thank", "try", "turn", "wait", "write"
        };

        /// <summary>
        ///     Least of the learner's own words a correction must keep for it to be a correction
        ///     of what they said rather than a different sentence put in its place.
        ///
        ///     <para>
        ///     Measured: over 512 recorded live turns, every correction of a real grammatical
        ///     error kept at least half the learner's words, while the corrections that replaced
        ///     the sentence outright kept a quarter or fewer. The threshold sits in that gap.
        ///     </para>
        /// </summary>
        public const float CorrectionRetentionFloor = 0.35f;

        /// <summary>A run of letters, apostrophes included so contractions count once.</summary>
        private static readonly Regex WordPattern = new Regex(@"[A-Za-z’']+", RegexOptions.Compiled);

        /// <summary>Splits the authored "Original: X -&gt; Corrected: Y" shape into its halves.</summary>
        private static readonly Regex CorrectionShape = new Regex(
            @"Original:\s*(.*?)\s*->\s*Corrected:\s*(.*)$",
            RegexOptions.Singleline | RegexOptions.IgnoreCase | RegexOptions.Compiled);

        /// <summary>Ellipses, quotes and sentence punctuation around a quoted phrase.</summary>
        private static readonly Regex QuotedPhraseEdges = new Regex(
            @"^[\s.…""'“”‘’]+|[\s.…!?,""'“”‘’]+$", RegexOptions.Compiled);

        /// <summary>Extracts a complete string value for the reply key from truncated JSON payloads.</summary>
        private static readonly Regex SalvageJsonReplyField = new Regex(
            @"""reply""\s*:\s*""((?:\\.|[^""\\])*)""",
            RegexOptions.Singleline | RegexOptions.Compiled);

        /// <summary>Prefix <see cref="ComposeFeedback"/> writes the correction under.</summary>
        private const string GrammarBullet = "• Grammar: ";

        /// <summary>
        ///     The line <see cref="ComposeFeedback"/> writes for a PERFECT turn with nothing to
        ///     correct. Named so a later rule that takes PERFECT away can take this line with it.
        /// </summary>
        private const string GrammarConfirmation = GrammarBullet + "Great! Your sentence is grammatically correct.";

        // ──────────────────── ENTRY POINT ────────────────────

        /// <summary>
        ///     Parses raw model output. Tries structured JSON first, then the legacy tag format.
        ///     Never throws and never returns null.
        /// </summary>
        public static ParsedResponse Parse(string rawResponse)
        {
            return Parse(rawResponse, null);
        }

        /// <summary>
        ///     As <see cref="Parse(string)"/>, but told what the learner actually said, so the
        ///     objective-credit rule can be decided from the utterance instead of only from the
        ///     score the model assigned to it. Pass null on any path where the client has no
        ///     independent copy of the utterance; the model's own transcription is used instead.
        /// </summary>
        public static ParsedResponse Parse(string rawResponse, string learnerUtterance)
        {
            if (string.IsNullOrWhiteSpace(rawResponse))
                return new ParsedResponse();

            ParsedResponse result = null;

            string candidate = StripCodeFence(rawResponse.Trim());
            if (candidate.StartsWith("{", StringComparison.Ordinal))
            {
                result = TryParseStructured(candidate);
                if (result == null &&
                    rawResponse.IndexOf("[SPOKEN]", StringComparison.OrdinalIgnoreCase) < 0 &&
                    rawResponse.IndexOf("[FEEDBACK]", StringComparison.OrdinalIgnoreCase) < 0 &&
                    rawResponse.IndexOf("[TRANSCRIPTION]", StringComparison.OrdinalIgnoreCase) < 0)
                {
                    Match salvage = SalvageJsonReplyField.Match(candidate);
                    string spokenText = "";
                    if (salvage.Success)
                    {
                        try { spokenText = Regex.Unescape(salvage.Groups[1].Value); }
                        catch (ArgumentException) { spokenText = salvage.Groups[1].Value; }
                    }
                    result = new ParsedResponse { Text = CollapseWhitespace(spokenText) };
                }
            }

            if (result == null)
                result = ParseTagged(rawResponse);

            // Score before credit, so both read the score the learner will actually see.
            ApplyFragmentCeiling(result, learnerUtterance);

            // The one place objective credit is decided, for either payload format. It used to
            // run at the end of each path, which is how the two paths came to disagree once.
            ApplyObjectiveCreditPolicy(result, learnerUtterance);
            ApplyCorrectionPolicy(result, learnerUtterance);

            return result;
        }

        // ──────────────────── STRUCTURED (PREFERRED) ────────────────────

        /// <summary>Returns null when the payload is not a usable <see cref="CoachTurn"/>.</summary>
        private static ParsedResponse TryParseStructured(string json)
        {
            CoachTurn turn;
            try
            {
                turn = JsonConvert.DeserializeObject<CoachTurn>(json);
            }
            catch (JsonException)
            {
                return null;
            }

            if (turn == null)
                return null;

            // A well-formed JSON object that is not a coaching turn (an API error body, say)
            // deserialises with every field null. Reject it so the tag parser gets a chance.
            if (string.IsNullOrWhiteSpace(turn.Reply))
                return null;

            var result = new ParsedResponse
            {
                Text = CollapseWhitespace(turn.Reply),
                UserTranscription = (turn.Transcription ?? "").Trim(),
                FeedbackStatus = NormalizeStatus(turn.Status),
                Correction = (turn.Correction ?? "").Trim(),
                PronunciationNote = (turn.Pronunciation ?? "").Trim(),
                Tip = (turn.Tip ?? "").Trim(),
                TipThai = (turn.TipThai ?? "").Trim(),
                Emotion = ParseEmotion(turn.Emotion),
                EmotionIntensity = Clamp(turn.EmotionIntensity, 1, 3)
            };

            if (turn.Score >= 0 && turn.Score <= 100)
            {
                result.Score = turn.Score;
                result.HasScore = true;
            }

            if (turn.Vocabulary >= 0 && turn.Vocabulary <= 100)
                result.VocabularyScore = turn.Vocabulary;

            result.ReplyThai = (turn.ReplyThai ?? "").Trim();

            if (turn.Suggestions != null)
            {
                foreach (Suggestion s in turn.Suggestions)
                {
                    string en = (s?.En ?? "").Trim();
                    if (en.Length == 0 || result.Suggestions.Count >= 3) continue;
                    result.Suggestions.Add(new Suggestion { En = en, Th = (s.Th ?? "").Trim() });
                }
            }

            result.HasCorrection = result.Correction.Length > 0;
            result.HasPronunciationNote = result.PronunciationNote.Length > 0;

            // The model regularly rewrites a sentence and still scores it 90+, which leaves the
            // learner reading "Fluent" directly above the correction of their own line. Hold the
            // number to the band the rubric defines; this only enforces consistency with the
            // feedback already on screen, it is not a second opinion on the English.
            if (result.HasCorrection && result.HasScore && result.Score >= FlawlessBandFloor)
                result.Score = FlawlessBandFloor - 1;
            result.FeedbackText = ComposeFeedback(turn);
            result.BetterPhrasing = (turn.BetterPhrasing ?? "").Trim();
            result.HasFeedback = !string.IsNullOrWhiteSpace(result.FeedbackText) || result.HasScore;

            if (turn.QuestsCompleted != null)
            {
                foreach (int q in turn.QuestsCompleted)
                    if (!result.CompletedQuests.Contains(q))
                        result.CompletedQuests.Add(q);
            }

            return result;
        }

        private static string ComposeFeedback(CoachTurn turn)
        {
            var sb = new StringBuilder();

            string correction = (turn.Correction ?? "").Trim();
            if (correction.Length > 0)
                sb.Append(GrammarBullet).Append(correction).Append('\n');
            else if (string.Equals(turn.Status, "PERFECT", StringComparison.OrdinalIgnoreCase))
                sb.Append(GrammarConfirmation).Append('\n');

            string pronunciation = (turn.Pronunciation ?? "").Trim();
            if (pronunciation.Length > 0)
                sb.Append("• Pronunciation: ").Append(pronunciation).Append('\n');

            string tip = (turn.Tip ?? "").Trim();
            if (tip.Length > 0)
                sb.Append("• Quick Tip: ").Append(tip).Append('\n');

            // Kept as an ASCII label with Thai only in the value, so the label survives any
            // downstream formatting that assumes Latin text.
            string tipThai = (turn.TipThai ?? "").Trim();
            if (tipThai.Length > 0)
                sb.Append("• Thai: ").Append(tipThai).Append('\n');

            string better = (turn.BetterPhrasing ?? "").Trim();
            if (better.Length > 0)
                sb.Append("• Better Phrasing: ").Append(better).Append('\n');

            return sb.ToString().TrimEnd();
        }

        // ──────────────────── LEGACY TAG FORMAT (FALLBACK) ────────────────────

        private static readonly Regex QuestTag =
            new Regex(@"\[QUEST_(?:DONE|COMPLETE):\s*([1-3])\s*\]", RegexOptions.IgnoreCase);

        private static readonly Regex TranscriptionBlock =
            new Regex(@"\[TRANSCRIPTION\](.*?)(?:\[/TRANSCRIPTION\]|(?=\[FEEDBACK\])|(?=\[SPOKEN\])|$)",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);

        private static readonly Regex FeedbackBlock =
            new Regex(@"\[FEEDBACK\](.*?)(?:\[/FEEDBACK\]|(?=\[SPOKEN\])|$)",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);

        private static readonly Regex SpokenBlock =
            new Regex(@"\[SPOKEN\](.*?)(?:\[/SPOKEN\]|$)",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);

        private static readonly Regex ScoreTag =
            new Regex(@"\[SCORE:\s*(-?\d{1,3})\s*\]", RegexOptions.IgnoreCase);

        private static readonly Regex StatusTag =
            new Regex(@"\[STATUS:\s*([A-Z_]+)\s*\]", RegexOptions.IgnoreCase);

        private static readonly Regex EmotionTag =
            new Regex(@"\[EMOTION:\s*([A-Za-z]+)\s*(?::\s*(\d)\s*)?\]", RegexOptions.IgnoreCase);

        private static ParsedResponse ParseTagged(string raw)
        {
            var result = new ParsedResponse();
            string working = raw;

            // ── Quest tags, anywhere in the payload ──
            foreach (Match m in QuestTag.Matches(working))
            {
                int q = int.Parse(m.Groups[1].Value);
                if (!result.CompletedQuests.Contains(q))
                    result.CompletedQuests.Add(q);
            }
            result.CompletedQuests.Sort();
            working = QuestTag.Replace(working, "");

            // ── Transcription block ──
            Match trans = TranscriptionBlock.Match(working);
            if (trans.Success)
            {
                result.UserTranscription = trans.Groups[1].Value.Trim();
                working = working.Remove(trans.Index, trans.Length);
            }

            // ── Feedback block ──
            Match feedback = FeedbackBlock.Match(working);
            if (feedback.Success)
            {
                string body = feedback.Groups[1].Value.Trim();
                working = working.Remove(feedback.Index, feedback.Length);

                Match score = ScoreTag.Match(body);
                if (score.Success && int.TryParse(score.Groups[1].Value, out int parsedScore))
                {
                    result.Score = Clamp(parsedScore, 0, 100);
                    result.HasScore = true;
                }

                result.FeedbackStatus = ReadStatus(body);

                // Strip the machine tags so they never reach the learner's screen.
                body = ScoreTag.Replace(body, "");
                body = StatusTag.Replace(body, "");
                body = TidyBullets(body);

                result.FeedbackText = body;
                result.HasFeedback = body.Length > 0 || result.HasScore;
            }

            // ── Spoken block ──
            Match spoken = SpokenBlock.Match(working);
            string content = spoken.Success ? spoken.Groups[1].Value : working;

            // ── Emotion tag (first one wins; all are stripped) ──
            Match emotion = EmotionTag.Match(content);
            if (emotion.Success)
            {
                result.Emotion = ParseEmotion(emotion.Groups[1].Value);
                result.EmotionIntensity = emotion.Groups[2].Success && int.TryParse(emotion.Groups[2].Value, out int i)
                    ? Clamp(i, 1, 3)
                    : 2;
            }
            content = EmotionTag.Replace(content, " ");

            // Anything left that still looks like a tag is machine chatter, not dialogue.
            content = CollapseWhitespace(content);

            // Deliberately NOT falling back to the raw payload here: doing so used to make
            // the avatar read its own markup aloud when a block failed to close.
            result.Text = content;

            return result;
        }

        /// <summary>
        ///     Single enforcement point for objective credit, applied by both wire formats.
        ///
        ///     <para>
        ///     An objective is a speaking goal, so it cannot be cleared by a turn that was not
        ///     a sentence. The model marks these generously - "total price" was enough to tick
        ///     "ask the total price" - which let the learner progress without ever practising
        ///     the thing the objective is asking for.
        ///     </para>
        ///
        ///     <para>
        ///     This lived inline in the structured path only, so a payload arriving in the
        ///     legacy tag format kept credit that the structured path would have stripped.
        ///     Keeping the rule in one place is what stops the two paths diverging again.
        ///     </para>
        /// </summary>
        /// <summary>
        ///     Holds a one- or two-word fragment to the rubric's 55 ceiling.
        ///
        ///     <para>
        ///     The model is told this rule and usually keeps it. gemini-3.5-flash kept it on all
        ///     64 fragments in a recorded run; gemini-3.5-flash-lite broke it on two of 64,
        ///     scoring "Painkiller" 100 - which puts a cheer on a single word and a perfect mark
        ///     in the learner's history. Like the correction clamp above, this enforces what the
        ///     rubric already says rather than offering a second opinion on the English, and it
        ///     means the rule no longer depends on which model is answering.
        ///     </para>
        ///
        ///     <para>
        ///     Only ever lowers, and only a score above the ceiling. Judged from the client's
        ///     own copy of the utterance where there is one; with no utterance at all it has no
        ///     opinion and changes nothing, as the objective rule does.
        ///     </para>
        /// </summary>
        private static void ApplyFragmentCeiling(ParsedResponse result, string learnerUtterance)
        {
            if (!result.HasScore || result.Score <= FragmentScoreCeiling)
                return;

            string utterance = string.IsNullOrWhiteSpace(learnerUtterance)
                ? result.UserTranscription
                : learnerUtterance;

            if (string.IsNullOrWhiteSpace(utterance) || !IsFragment(utterance))
                return;

            result.Score = FragmentScoreCeiling;

            // A capped fragment labelled Perfect would contradict its own number, and so would
            // the confirmation composed from that label before this rule ran.
            if (result.FeedbackStatus == "PERFECT")
            {
                result.FeedbackStatus = "GOOD";
                result.FeedbackText = WithoutConfirmationLine(result.FeedbackText);
                result.HasFeedback = !string.IsNullOrWhiteSpace(result.FeedbackText) || result.HasScore;
            }
        }

        /// <summary>
        ///     Whether an utterance is too short to be a sentence: two words or fewer, with no
        ///     contraction and no imperative verb among them.
        ///
        ///     <para>
        ///     The contraction test is what keeps "I'm fine." and "It's okay." out. They are two
        ///     words by this count - apostrophes are part of a word here - and complete sentences,
        ///     and marking a learner down for "I'm fine." would be a worse fault than the one this
        ///     prevents. A possessive ("Tom's bag") also passes, and simply keeps the model's score.
        ///     </para>
        /// </summary>
        public static bool IsFragment(string utterance)
        {
            MatchCollection words = WordPattern.Matches(utterance ?? "");
            if (words.Count == 0 || words.Count > FragmentMaxWords)
                return false;

            foreach (Match w in words)
            {
                if (w.Value.IndexOf('\'') >= 0 || w.Value.IndexOf('’') >= 0)
                    return false;
                if (ImperativeVerbs.Contains(w.Value))
                    return false;
            }

            return true;
        }

        private static void ApplyObjectiveCreditPolicy(ParsedResponse result, string learnerUtterance)
        {
            if (result.CompletedQuests.Count == 0)
                return;

            // Gate one, on the score. Kept, but it is not sufficient on its own: the score is
            // the model's own opinion, and across two recorded runs the same fragment scored 55
            // and then 40. Anything the model rounds up to the band floor passes this gate.
            bool scoreAllows = !result.HasScore || result.Score >= SentenceBandFloor;

            // Gate two, on what the learner said. Decided from the client's own copy of the
            // utterance where there is one, so the rule does not rest entirely on a number the
            // model chose. Fragments such as "Recommend" and "No allergy" were observed clearing
            // objectives through gate one alone, which is the practice the objective asks for
            // not happening.
            string utterance = string.IsNullOrWhiteSpace(learnerUtterance)
                ? result.UserTranscription
                : learnerUtterance;

            // No utterance to judge means no opinion: an absent transcription must not silently
            // strip credit the learner earned.
            bool utteranceAllows = string.IsNullOrWhiteSpace(utterance)
                                   || IsSubstantialUtterance(utterance);

            if (!scoreAllows || !utteranceAllows)
            {
                result.CompletedQuests.Clear();
                return;
            }

            result.CompletedQuests.RemoveAll(q => q < 1 || q > MaxObjectiveId);
        }

        /// <summary>
        ///     Drops a correction that is not a correction of what the learner said.
        ///
        ///     <para>
        ///     The contract asks for "Original: X -&gt; Corrected: Y", meaning the learner's line
        ///     and a repaired version of it. The model also uses the field for something else:
        ///     when the learner says something well formed but off-topic it puts a different
        ///     utterance there, and twice it put the avatar's own line there - a learner who
        ///     wrote a correct sentence was shown "You haven't ordered anything yet" as the
        ///     thing they should have said. The panel labels this field Grammar, so a learner
        ///     reads their own correct English struck through and replaced.
        ///     </para>
        ///
        ///     <para>
        ///     Expanding a fragment into a full sentence is the opposite case and must survive:
        ///     answering "Wifi" with "What is the Wi-Fi password, please?" retains none of the
        ///     original and is exactly the teaching this application exists to do. The two are
        ///     told apart by whether the learner produced a sentence in the first place.
        ///     </para>
        ///
        ///     <para>
        ///     The explanation in the tip is left alone. It is accurate in all of these turns,
        ///     and it is where the real feedback lives once the false correction is gone.
        ///     </para>
        /// </summary>
        private static void ApplyCorrectionPolicy(ParsedResponse result, string learnerUtterance)
        {
            if (!result.HasCorrection)
                return;

            string utterance = string.IsNullOrWhiteSpace(learnerUtterance)
                ? result.UserTranscription
                : learnerUtterance;

            // Nothing to compare against, or the learner did not manage a sentence: in both
            // cases a rewrite is either unjudgeable or is the legitimate expansion of a
            // fragment, so it stands.
            if (string.IsNullOrWhiteSpace(utterance) || !IsSubstantialUtterance(utterance))
                return;

            Match shape = CorrectionShape.Match(result.Correction);
            if (!shape.Success)
                return;

            if (AnchoredRetention(utterance, shape.Groups[1].Value, shape.Groups[2].Value)
                >= CorrectionRetentionFloor)
                return;

            result.Correction = "";
            result.HasCorrection = false;
            result.FeedbackText = WithoutGrammarLine(result.FeedbackText);
            result.HasFeedback = !string.IsNullOrWhiteSpace(result.FeedbackText) || result.HasScore;
        }

        /// <summary>
        ///     How much of the learner's utterance survives the correction once the correction is
        ///     applied to that utterance.
        ///
        ///     <para>
        ///     The model often quotes only the wrong phrase - "more bad -&gt; worse", "come -&gt;
        ///     coming" - and measured within the phrase alone a one-word repair keeps nothing, so
        ///     genuine corrections were hidden. When the quoted original occurs in the utterance
        ///     as whole words it is replaced there and retention is measured over the whole
        ///     utterance. When it does not occur - a placeholder the model invented, or a line
        ///     the learner never said - the phrase-level measure is used as before.
        ///     </para>
        /// </summary>
        private static float AnchoredRetention(string utterance, string original, string corrected)
        {
            string o = TrimQuoted(original).ToLowerInvariant();
            string c = TrimQuoted(corrected).ToLowerInvariant();
            string u = " " + JoinWords(utterance).ToLowerInvariant() + " ";
            string needle = " " + JoinWords(o) + " ";

            int at = needle.Trim().Length == 0 ? -1 : u.IndexOf(needle, StringComparison.Ordinal);
            if (at < 0)
                return RetainedFraction(original, corrected);

            string applied = u.Substring(0, at) + " " + c + " " + u.Substring(at + needle.Length);
            return RetainedFraction(utterance, applied);
        }

        /// <summary>Strips surrounding ellipses, quotes and sentence punctuation from a quoted phrase.</summary>
        private static string TrimQuoted(string s)
        {
            return QuotedPhraseEdges.Replace(s ?? "", "");
        }

        /// <summary>The words of <paramref name="s"/>, single-spaced.</summary>
        private static string JoinWords(string s)
        {
            var words = new List<string>();
            foreach (Match m in WordPattern.Matches(s ?? ""))
                words.Add(m.Value);
            return string.Join(" ", words.ToArray());
        }

        /// <summary>How much of <paramref name="original"/> survives into <paramref name="corrected"/>.</summary>
        private static float RetainedFraction(string original, string corrected)
        {
            var before = new HashSet<string>();
            foreach (Match m in WordPattern.Matches(original ?? ""))
                before.Add(m.Value.ToLowerInvariant());

            if (before.Count == 0)
                return 1f;

            var after = new HashSet<string>();
            foreach (Match m in WordPattern.Matches(corrected ?? ""))
                after.Add(m.Value.ToLowerInvariant());

            int kept = 0;
            foreach (string w in before)
                if (after.Contains(w))
                    kept++;

            return (float)kept / before.Count;
        }

        /// <summary>
        ///     Removes the grammar line from an already-composed feedback block, leaving the
        ///     tip, the pronunciation note and the rest in place.
        /// </summary>
        private static string WithoutGrammarLine(string feedback)
        {
            if (string.IsNullOrEmpty(feedback))
                return feedback;

            var kept = new List<string>();
            foreach (string line in feedback.Split('\n'))
                if (!line.StartsWith(GrammarBullet, StringComparison.Ordinal))
                    kept.Add(line);

            return string.Join("\n", kept.ToArray()).Trim();
        }

        /// <summary>
        ///     Removes the PERFECT confirmation from an already-composed feedback block. Matches
        ///     the whole line, so a real correction under the same Grammar bullet is never touched.
        /// </summary>
        private static string WithoutConfirmationLine(string feedback)
        {
            if (string.IsNullOrEmpty(feedback))
                return feedback;

            var kept = new List<string>();
            foreach (string line in feedback.Split('\n'))
                if (!string.Equals(line, GrammarConfirmation, StringComparison.Ordinal))
                    kept.Add(line);

            return string.Join("\n", kept.ToArray()).Trim();
        }

        /// <summary>
        ///     Whether an utterance is long enough to count as practising an objective.
        ///
        ///     <para>
        ///     This is deliberately a length test and not a grammatical one. Learner English is
        ///     often ungrammatical by definition, so a parser would reject the very turns the
        ///     application exists to accept; length separates a bare noun phrase from an attempt
        ///     at a sentence without judging the attempt.
        ///     </para>
        /// </summary>
        public static bool IsSubstantialUtterance(string utterance)
        {
            // Japanese has no spaces and no Latin letters, so it counted as no words at all and
            // could never complete an objective. Its kana and kanji are counted instead.
            int japanese = CountJapaneseCharacters(utterance);
            if (japanese > 0)
                return japanese >= MinimumObjectiveJapaneseCharacters;

            return CountWords(utterance) >= MinimumObjectiveWords;
        }

        /// <summary>
        ///     Kana and kanji needed for a Japanese turn to count as practising an objective: six
        ///     keeps 「お会計」 or 「ラテ」 out and lets 「ラテをください」 in. The model's own
        ///     rule - no predicate, no objective - does the rest.
        /// </summary>
        public const int MinimumObjectiveJapaneseCharacters = 6;

        /// <summary>Hiragana, katakana (with the long-vowel mark) and kanji in a string.</summary>
        public static int CountJapaneseCharacters(string text)
        {
            if (string.IsNullOrEmpty(text)) return 0;

            int n = 0;
            foreach (char c in text)
            {
                if ((c >= '\u3041' && c <= '\u309F') ||   // hiragana
                    (c >= '\u30A0' && c <= '\u30FF') ||   // katakana, ー
                    (c >= '\u4E00' && c <= '\u9FFF') ||   // common kanji
                    (c >= '\u3400' && c <= '\u4DBF'))     // rarer kanji
                    n++;
            }
            return n;
        }

        /// <summary>Words in an utterance. Apostrophes bind, so "I'll" counts once.</summary>
        public static int CountWords(string utterance)
        {
            if (string.IsNullOrWhiteSpace(utterance))
                return 0;

            return WordPattern.Matches(utterance).Count;
        }

        private static string ReadStatus(string feedbackBody)
        {
            Match tag = StatusTag.Match(feedbackBody);
            if (tag.Success)
                return NormalizeStatus(tag.Groups[1].Value);

            // No explicit tag: fall back to keywords. NEEDS_FIX is tested first because
            // "almost perfect, but here is a correction" must not read as PERFECT.
            if (Contains(feedbackBody, "NEEDS_FIX") || Contains(feedbackBody, "CORRECTION") ||
                Contains(feedbackBody, "[Fix]") || feedbackBody.Contains("❌"))
                return "NEEDS_FIX";

            if (Contains(feedbackBody, "PERFECT") || Contains(feedbackBody, "EXCELLENT"))
                return "PERFECT";

            return "GOOD";
        }

        // ──────────────────── SHARED HELPERS ────────────────────

        internal static AvatarEmotion ParseEmotion(string value)
        {
            switch ((value ?? "").Trim().ToLowerInvariant())
            {
                case "happy":
                case "joy":
                    return AvatarEmotion.Happy;
                case "sad":
                case "sadness":
                    return AvatarEmotion.Sad;
                case "angry":
                case "anger":
                    return AvatarEmotion.Angry;
                case "surprised":
                case "surprise":
                    return AvatarEmotion.Surprised;
                case "fear":
                case "scared":
                    return AvatarEmotion.Fear;
                case "disgust":
                    return AvatarEmotion.Disgust;
                case "thinking":
                case "curious":
                    return AvatarEmotion.Thinking;
                default:
                    return AvatarEmotion.Neutral;
            }
        }

        internal static string NormalizeStatus(string value)
        {
            string v = (value ?? "").Trim().ToUpperInvariant();
            switch (v)
            {
                case "PERFECT":
                case "EXCELLENT":
                    return "PERFECT";
                case "NEEDS_FIX":
                case "NEEDSFIX":
                case "NEEDS FIX":
                    return "NEEDS_FIX";
                default:
                    return "GOOD";
            }
        }

        /// <summary>Removes a leading ```json / ``` fence and its closing fence, if present.</summary>
        internal static string StripCodeFence(string s)
        {
            if (!s.StartsWith("```", StringComparison.Ordinal))
                return s;

            int firstNewline = s.IndexOf('\n');
            if (firstNewline < 0)
                return s;

            string body = s.Substring(firstNewline + 1);
            int closing = body.LastIndexOf("```", StringComparison.Ordinal);
            if (closing >= 0)
                body = body.Substring(0, closing);

            return body.Trim();
        }

        private static bool Contains(string haystack, string needle)
        {
            return haystack.IndexOf(needle, StringComparison.OrdinalIgnoreCase) >= 0;
        }

        /// <summary>Collapses runs of whitespace to single spaces and trims. Empty in, empty out.</summary>
        private static string CollapseWhitespace(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return "";
            return Regex.Replace(s, @"\s+", " ").Trim();
        }

        /// <summary>Drops bullet lines left empty once their tags were stripped.</summary>
        private static string TidyBullets(string body)
        {
            var kept = new List<string>();
            foreach (string line in body.Split('\n'))
            {
                string trimmed = line.TrimEnd();
                string bare = trimmed.TrimStart(' ', '\t', '•', '-', '*');
                if (bare.Trim().Length == 0) continue;
                // "• Score:" with nothing after it carries no information once [SCORE:n] is gone.
                if (Regex.IsMatch(bare, @"^(Score|Status)\s*:\s*$", RegexOptions.IgnoreCase)) continue;
                kept.Add(trimmed);
            }
            return string.Join("\n", kept).Trim();
        }

        private static int Clamp(int value, int min, int max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }
    }
}
