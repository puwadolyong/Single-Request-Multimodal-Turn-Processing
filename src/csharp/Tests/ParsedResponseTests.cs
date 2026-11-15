using NUnit.Framework;

namespace GeminiAvatar.Tests
{
    /// <summary>
    ///     Contract tests for <see cref="ParsedResponse.Parse"/>.
    ///
    ///     These exist because the parser sits between an unpredictable model and the
    ///     learner's recorded score history: a formatting slip must degrade visibly,
    ///     never invent data.
    /// </summary>
    [TestFixture]
    public class ParsedResponseTests
    {
        // ──────────────────── STRUCTURED JSON PATH ────────────────────

        private const string FullJson = @"{
            ""transcription"": ""I want a coffee please"",
            ""score"": 78,
            ""status"": ""NEEDS_FIX"",
            ""correction"": ""'I'd like a coffee, please.'"",
            ""tip"": ""'I'd like' is more polite than 'I want' when ordering."",
            ""betterPhrasing"": ""Could I get a latte, please?"",
            ""reply"": ""Of course! Would you like that hot or iced?"",
            ""emotion"": ""happy"",
            ""emotionIntensity"": 2,
            ""questsCompleted"": [1]
        }";

        [Test]
        public void StructuredJson_MapsEveryField()
        {
            var r = ParsedResponse.Parse(FullJson);

            Assert.AreEqual("Of course! Would you like that hot or iced?", r.Text);
            Assert.AreEqual("I want a coffee please", r.UserTranscription);
            // Kept apart from the display copy, which is rich text with colour tags in
            // it and cannot be handed to a speech synthesiser.
            Assert.AreEqual("Could I get a latte, please?", r.BetterPhrasing);
            Assert.AreEqual("NEEDS_FIX", r.FeedbackStatus);
            Assert.AreEqual(78, r.Score);
            Assert.IsTrue(r.HasScore);
            Assert.IsTrue(r.HasFeedback);
            Assert.AreEqual(AvatarEmotion.Happy, r.Emotion);
            Assert.AreEqual(2, r.EmotionIntensity);
            CollectionAssert.AreEqual(new[] { 1 }, r.CompletedQuests);
        }

        [Test]
        public void StructuredJson_ComposesReadableFeedback()
        {
            var r = ParsedResponse.Parse(FullJson);

            StringAssert.Contains("Grammar:", r.FeedbackText);
            StringAssert.Contains("Quick Tip:", r.FeedbackText);
            StringAssert.Contains("Better Phrasing:", r.FeedbackText);
            // Machine fields must never leak into what the learner reads.
            StringAssert.DoesNotContain("[SCORE:", r.FeedbackText);
            StringAssert.DoesNotContain("[STATUS:", r.FeedbackText);
        }

        // ──────────────────── PRONUNCIATION & THAI ────────────────────

        [Test]
        public void PronunciationNoteIsSurfacedAndFlagged()
        {
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Sure thing!"",
                ""pronunciation"": ""'\''three'\'' came out as '\''tree'\'' - put your tongue between your teeth for the th sound."",
                ""emotion"": ""happy""
            }".Replace("'\''", "'"));

            Assert.IsTrue(r.HasPronunciationNote);
            StringAssert.Contains("tongue between your teeth", r.PronunciationNote);
            StringAssert.Contains("Pronunciation:", r.FeedbackText);
        }

        [Test]
        public void CorrectionIsFlaggedEvenOnAHighScore()
        {
            // The case the review list used to miss entirely: the coach rewrites the sentence
            // but still scores it well, so a score threshold alone never catches it.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Of course!"",
                ""score"": 92,
                ""status"": ""GOOD"",
                ""correction"": ""Original: Do you have Wi-Fi password? -> Corrected: Do you have the Wi-Fi password?"",
                ""emotion"": ""happy""
            }");

            Assert.IsTrue(r.HasCorrection, "a rewritten sentence must be flagged regardless of score");
            StringAssert.Contains("the Wi-Fi password", r.Correction);
            // 92 came back from the model, but a corrected line is held below the flawless band.
            Assert.AreEqual(ParsedResponse.FlawlessBandFloor - 1, r.Score);
        }

        [Test]
        public void ACorrectedSentenceCannotKeepAFlawlessScore()
        {
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Sure!"",
                ""score"": 96,
                ""status"": ""GOOD"",
                ""correction"": ""Original: I like an ice latte -> Corrected: I would like an iced latte"",
                ""emotion"": ""happy""
            }");

            Assert.IsTrue(r.HasCorrection);
            Assert.Less(r.Score, ParsedResponse.FlawlessBandFloor,
                "a rewritten sentence must not read as flawless");
        }

        [Test]
        public void AnUncorrectedSentenceKeepsItsHighScore()
        {
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Perfect!"", ""score"": 98, ""status"": ""PERFECT"", ""correction"": """", ""emotion"": ""happy"" }");

            Assert.AreEqual(98, r.Score, "a clean line must not be dragged down by the guard");
        }

        [Test]
        public void NoCorrection_IsNotFlagged()
        {
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Great!"", ""score"": 98, ""status"": ""PERFECT"", ""correction"": """", ""emotion"": ""happy"" }");

            Assert.IsFalse(r.HasCorrection);
            Assert.AreEqual("", r.Correction);
        }

        [Test]
        public void NoPronunciationNote_IsNotFlagged()
        {
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Great!"", ""pronunciation"": """", ""emotion"": ""happy"" }");

            Assert.IsFalse(r.HasPronunciationNote, "a typed turn has no audio to judge");
            Assert.AreEqual("", r.PronunciationNote);
            StringAssert.DoesNotContain("Pronunciation:", r.FeedbackText);
        }

        [Test]
        public void ThaiExplanationSurvivesTheRoundTrip()
        {
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Of course!"",
                ""tip"": ""Use 'would like' to sound polite."",
                ""tipThai"": ""ใช้ would like เพื่อให้สุภาพขึ้น แทนที่จะพูดว่า want"",
                ""emotion"": ""happy""
            }");

            Assert.AreEqual("ใช้ would like เพื่อให้สุภาพขึ้น แทนที่จะพูดว่า want", r.TipThai);
            StringAssert.Contains("• Thai: ", r.FeedbackText);
            StringAssert.Contains("สุภาพ", r.FeedbackText);
            // The label stays ASCII so downstream formatting can key off it.
            StringAssert.Contains("• Quick Tip: ", r.FeedbackText);
        }

        [Test]
        public void ThaiOffLeavesNoEmptySection()
        {
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Nice!"", ""tip"": ""Past tense here."", ""emotion"": ""happy"" }");

            Assert.AreEqual("", r.TipThai);
            StringAssert.DoesNotContain("Thai:", r.FeedbackText);
        }

        [Test]
        public void FeedbackSectionsKeepTeachingOrder()
        {
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Got it!"",
                ""correction"": ""Original: I go -> Corrected: I went"",
                ""pronunciation"": ""went, not vent"",
                ""tip"": ""Use past tense."",
                ""tipThai"": ""ใช้รูปอดีต"",
                ""betterPhrasing"": ""I headed over there."",
                ""emotion"": ""happy""
            }");

            string f = r.FeedbackText;
            Assert.Less(f.IndexOf("Grammar:"), f.IndexOf("Pronunciation:"));
            Assert.Less(f.IndexOf("Pronunciation:"), f.IndexOf("Quick Tip:"));
            Assert.Less(f.IndexOf("Quick Tip:"), f.IndexOf("Thai:"), "the Thai line should follow the tip it restates");
            Assert.Less(f.IndexOf("Thai:"), f.IndexOf("Better Phrasing:"));
        }

        [Test]
        public void StructuredJson_SurvivesMarkdownCodeFence()
        {
            var r = ParsedResponse.Parse("```json\n" + FullJson + "\n```");

            Assert.AreEqual("Of course! Would you like that hot or iced?", r.Text);
            Assert.AreEqual(78, r.Score);
        }

        [Test]
        public void StructuredJson_MissingScore_IsNotInvented()
        {
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Nice to meet you!"", ""status"": ""GOOD"", ""emotion"": ""happy"" }");

            Assert.IsFalse(r.HasScore, "an unscored turn must not report a score");
            Assert.AreEqual(0, r.Score);
            Assert.AreEqual("Nice to meet you!", r.Text);
        }

        [Test]
        public void StructuredJson_OutOfRangeScore_IsRejectedNotClamped()
        {
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Hi!"", ""score"": 250, ""status"": ""GOOD"", ""emotion"": ""neutral"" }");

            Assert.IsFalse(r.HasScore);
            Assert.AreEqual(0, r.Score);
        }

        [Test]
        public void StructuredJson_ClampsEmotionIntensity()
        {
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Wow!"", ""emotion"": ""surprised"", ""emotionIntensity"": 9 }");

            Assert.AreEqual(AvatarEmotion.Surprised, r.Emotion);
            Assert.AreEqual(3, r.EmotionIntensity);
        }

        [Test]
        public void StructuredJson_IgnoresQuestNumbersOutOfRange()
        {
            // Up to ParsedResponse.MaxObjectiveId (10, the job interview) is an objective; 0 and
            // 11 are not. Each scenario's own count is the controller's to enforce.
            var r = ParsedResponse.Parse(@"{ ""reply"": ""Great!"", ""questsCompleted"": [0, 2, 2, 7, 10, 11] }");

            CollectionAssert.AreEqual(new[] { 2, 7, 10 }, r.CompletedQuests);
        }

        [Test]
        public void AFragmentCannotClearAnObjective()
        {
            // "Total price" scored 40: it names a topic rather than asking anything, so it must
            // not tick the "ask the total price" objective.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""You haven't ordered anything yet - what can I get you?"",
                ""score"": 40,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: Total price -> Corrected: What is the total price?"",
                ""questsCompleted"": [3],
                ""emotion"": ""neutral""
            }");

            CollectionAssert.IsEmpty(r.CompletedQuests, "a turn below the sentence band earns no objective");
        }

        [Test]
        public void ACorrectionThatReplacesTheSentenceOutrightIsDropped()
        {
            // Recorded live turn. The learner's English was fine; they had answered a different
            // question from the one asked, and the model put the avatar's own line in the
            // correction field, which the panel labels Grammar.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""You haven't ordered anything yet - what can I get you?"",
                ""score"": 50,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: What was the total for my order? -> Corrected: You haven't ordered anything yet - what can I get you?"",
                ""tip"": ""You cannot ask for the total before placing an order."",
                ""emotion"": ""neutral""
            }", "What was the total for my order?");

            Assert.IsFalse(r.HasCorrection, "this is not a correction of what the learner said");
            Assert.IsEmpty(r.Correction);
            StringAssert.DoesNotContain("Grammar:", r.FeedbackText);
        }

        [Test]
        public void TheExplanationSurvivesWhenTheCorrectionIsDropped()
        {
            // Dropping the false correction must not take the real feedback with it: the tip
            // is accurate on exactly these turns and is the only lesson left.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""I need to know about your return travel."",
                ""score"": 72,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: I have nothing to declare, only personal belongings. -> Corrected: Yes, I have my return ticket right here."",
                ""tip"": ""Answer the specific question the officer asked."",
                ""emotion"": ""neutral""
            }", "I have nothing to declare, only personal belongings.");

            Assert.IsFalse(r.HasCorrection);
            StringAssert.Contains("Answer the specific question", r.FeedbackText);
            Assert.IsTrue(r.HasFeedback);
        }

        [Test]
        public void ExpandingAFragmentIntoASentenceIsNotDropped()
        {
            // The opposite case, and the one that must survive: the learner managed a single
            // word and the model showed them the whole sentence. It retains none of the
            // original by construction, and it is the teaching the application exists to do.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Of course, one moment."",
                ""score"": 40,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: Wifi -> Corrected: What is the Wi-Fi password, please?"",
                ""tip"": ""A single word is not a complete sentence."",
                ""emotion"": ""neutral""
            }", "Wifi");

            Assert.IsTrue(r.HasCorrection, "expanding a fragment is legitimate coaching");
            StringAssert.Contains("Wi-Fi password", r.FeedbackText);
        }

        [Test]
        public void AGenuineGrammarCorrectionIsNotDropped()
        {
            // Retains most of the learner's words, which is what a repair looks like. Across
            // the recorded runs no correction of a real error kept less than half.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Certainly, to go it is."",
                ""score"": 76,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: I will taking it for takeout. -> Corrected: I will take it to go."",
                ""tip"": ""After 'will', use the base form of the verb."",
                ""emotion"": ""happy""
            }", "I will taking it for takeout.");

            Assert.IsTrue(r.HasCorrection);
            StringAssert.Contains("Grammar:", r.FeedbackText);
        }

        [Test]
        public void ACorrectionQuotingSomethingTheLearnerNeverSaidIsDropped()
        {
            // Recorded live turn: the model invented a placeholder for the original.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Tell me your symptoms first."",
                ""score"": 55,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: (No symptoms described) -> Corrected: I have a bad cough and a fever that started two days ago."",
                ""tip"": ""Describe your symptoms and how long you have had them."",
                ""emotion"": ""neutral""
            }", "What did the previous doctor write in my file?");

            Assert.IsFalse(r.HasCorrection);
        }

        [Test]
        public void ACorrectionStandsWhenThereIsNoUtteranceToJudgeItAgainst()
        {
            // No copy of the utterance from either side. The rule must not drop feedback it
            // cannot justify dropping.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Got it."",
                ""score"": 70,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: something -> Corrected: something else entirely"",
                ""emotion"": ""neutral""
            }");

            Assert.IsTrue(r.HasCorrection);
        }

        [TestCase("more bad", "worse", "The pain is more bad than yesterday.")]
        [TestCase("come", "coming", "I am come to the party tonight.")]
        [TestCase("reservate", "reservation", "I want to make a reservate for two people.")]
        public void APhraseLevelCorrectionIsNotDropped(string original, string corrected, string utterance)
        {
            // The model quoted only the wrong phrase. Measured within the phrase, a one-word
            // repair keeps none of it; applied to the utterance, it keeps nearly all of it.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""I see."",
                ""score"": 70,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: " + original + @" -> Corrected: " + corrected + @""",
                ""tip"": ""Small grammar fix."",
                ""emotion"": ""neutral""
            }", utterance);

            Assert.IsTrue(r.HasCorrection, "a phrase-level grammar fix is a correction of what was said");
            StringAssert.Contains("Grammar:", r.FeedbackText);
        }

        [Test]
        public void AWholeSentenceReplacementIsStillDroppedWhenQuotedVerbatim()
        {
            // The quoted original is the learner's whole line, so anchoring on the utterance
            // changes nothing: the avatar's line put in its place keeps almost none of it.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""You haven't ordered anything yet."",
                ""score"": 50,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: Could you tell me the total price, please? -> Corrected: You haven't ordered anything yet."",
                ""emotion"": ""neutral""
            }", "Could you tell me the total price, please?");

            Assert.IsFalse(r.HasCorrection);
            StringAssert.DoesNotContain("Grammar:", r.FeedbackText);
        }

        [Test]
        public void AnEllipsisQuotedClauseIsMatchedInsideTheUtterance()
        {
            // Leading ellipsis and trailing full stop are trimmed before the quote is looked up.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""It is a lovely spot."",
                ""score"": 72,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: ...because there is quiet. -> Corrected: ...because it is quiet."",
                ""emotion"": ""happy""
            }", "I like this cafe because there is quiet.");

            Assert.IsTrue(r.HasCorrection);
        }

        [Test]
        public void AFragmentScoredInsideTheBandStillCannotClearAnObjective()
        {
            // Taken from a recorded live turn. The model scored the bare fragment "No allergy"
            // at 55, above the band floor, and credited the objective anyway: the score gate
            // alone lets this through, which is why the utterance is now tested as well.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Good to know. Let me note that down."",
                ""transcription"": ""No allergy"",
                ""score"": 55,
                ""status"": ""NEEDS_FIX"",
                ""correction"": ""Original: No allergy -> Corrected: I do not have any drug allergies."",
                ""questsCompleted"": [2],
                ""emotion"": ""neutral""
            }", "No allergy");

            CollectionAssert.IsEmpty(r.CompletedQuests,
                "a two-word fragment earns no objective however the model scored it");
        }

        [Test]
        public void TheLearnersOwnWordsOutrankTheModelsTranscription()
        {
            // The transcription is a sentence but the learner typed a fragment. On the text
            // path the client knows what was actually sent, and that is what decides.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Certainly."",
                ""transcription"": ""Could you please tell me the total price?"",
                ""score"": 92,
                ""status"": ""GOOD"",
                ""questsCompleted"": [3],
                ""emotion"": ""happy""
            }", "Total price");

            CollectionAssert.IsEmpty(r.CompletedQuests);
        }

        [Test]
        public void TheTranscriptionDecidesWhenTheClientHasNoCopyOfTheUtterance()
        {
            // The speech path: recognition happens inside the same request, so the only
            // record of what was said is the one the response carries.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Of course."",
                ""transcription"": ""Turn left"",
                ""score"": 55,
                ""status"": ""NEEDS_FIX"",
                ""questsCompleted"": [3],
                ""emotion"": ""neutral""
            }");

            CollectionAssert.IsEmpty(r.CompletedQuests);
        }

        [Test]
        public void AShortButCompleteSentenceStillClearsItsObjective()
        {
            // Four words is the floor, and it has to admit a real answer that happens to be
            // brief - the rule must not punish concision, only fragments.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Noted, thank you."",
                ""score"": 82,
                ""status"": ""GOOD"",
                ""questsCompleted"": [2],
                ""emotion"": ""happy""
            }", "I have no allergies");

            CollectionAssert.AreEqual(new[] { 2 }, r.CompletedQuests);
        }

        [Test]
        public void CreditSurvivesWhenNeitherTheClientNorTheModelReportsTheUtterance()
        {
            // No copy of the utterance from either side means no opinion about it. The rule
            // must not strip credit it cannot justify stripping.
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""Great!"",
                ""score"": 88,
                ""status"": ""GOOD"",
                ""questsCompleted"": [1],
                ""emotion"": ""happy""
            }");

            CollectionAssert.AreEqual(new[] { 1 }, r.CompletedQuests);
        }

        [Test]
        public void ContractionsCountAsOneWord()
        {
            Assert.AreEqual(4, ParsedResponse.CountWords("I'll take the muffin"));
            Assert.AreEqual(0, ParsedResponse.CountWords(""));
            Assert.AreEqual(0, ParsedResponse.CountWords(null));
            Assert.AreEqual(2, ParsedResponse.CountWords("  No   allergy!  "));
        }

        [Test]
        public void TheFragmentRuleHoldsOnTheLegacyTagFormatToo()
        {
            // The defect this rule was extracted to fix was that the two payload formats
            // disagreed. They are now decided at one point, so they cannot.
            var r = ParsedResponse.Parse(
                "[TRANSCRIPTION]No allergy[/TRANSCRIPTION]" +
                "[FEEDBACK][SCORE:55]Fix it.[/FEEDBACK]" +
                "[QUEST_DONE:2][SPOKEN]Noted.[/SPOKEN]");

            CollectionAssert.IsEmpty(r.CompletedQuests);
        }

        [Test]
        public void ARealSentenceStillClearsItsObjective()
        {
            var r = ParsedResponse.Parse(@"{
                ""reply"": ""That comes to six fifty."",
                ""score"": 88,
                ""status"": ""GOOD"",
                ""questsCompleted"": [3],
                ""emotion"": ""happy""
            }");

            CollectionAssert.AreEqual(new[] { 3 }, r.CompletedQuests);
        }

        [Test]
        public void QuestsSurviveWhenTheTurnWasNotScored()
        {
            // The legacy tag format often carries no score; that must not silently block progress.
            var r = ParsedResponse.Parse("[QUEST_DONE:2][SPOKEN]Nice one![/SPOKEN]");

            CollectionAssert.AreEqual(new[] { 2 }, r.CompletedQuests);
        }

        [Test]
        public void UnrelatedJsonObject_FallsThroughInsteadOfReturningBlank()
        {
            // An API error body is valid JSON but is not a coaching turn.
            var r = ParsedResponse.Parse(@"{ ""error"": { ""code"": 429, ""message"": ""quota exceeded"" } }");

            Assert.IsFalse(r.HasScore);
            Assert.IsFalse(r.HasFeedback);
        }

        // ──────────────────── LEGACY TAG PATH ────────────────────

        private const string LegacyTagged =
            "[TRANSCRIPTION]\nI want a coffee please\n[/TRANSCRIPTION]\n\n" +
            "[FEEDBACK]\n" +
            "• Score: [SCORE:78]\n" +
            "• Status: [STATUS:NEEDS_FIX]\n" +
            "• Grammar: Original: 'I want a coffee' -> Corrected: 'I'd like a coffee'\n" +
            "• Quick Tip: Use 'I'd like' when ordering.\n" +
            "[/FEEDBACK]\n\n" +
            "[SPOKEN]\n[EMOTION:happy:2] Of course! Hot or iced?\n[/SPOKEN]";

        [Test]
        public void LegacyTags_StillParse()
        {
            var r = ParsedResponse.Parse(LegacyTagged);

            Assert.AreEqual("Of course! Hot or iced?", r.Text);
            Assert.AreEqual("I want a coffee please", r.UserTranscription);
            Assert.AreEqual(78, r.Score);
            Assert.IsTrue(r.HasScore);
            Assert.AreEqual("NEEDS_FIX", r.FeedbackStatus);
            Assert.AreEqual(AvatarEmotion.Happy, r.Emotion);
            Assert.AreEqual(2, r.EmotionIntensity);
        }

        [Test]
        public void LegacyTags_StripMachineTagsFromDisplayedFeedback()
        {
            var r = ParsedResponse.Parse(LegacyTagged);

            StringAssert.DoesNotContain("[SCORE:", r.FeedbackText);
            StringAssert.DoesNotContain("[STATUS:", r.FeedbackText);
            StringAssert.Contains("Quick Tip:", r.FeedbackText);
        }

        [Test]
        public void LegacyTags_MissingScoreTag_IsNotInvented()
        {
            var r = ParsedResponse.Parse(
                "[FEEDBACK]\n• Grammar: Looks good.\n[/FEEDBACK]\n[SPOKEN]Nice one![/SPOKEN]");

            Assert.IsFalse(r.HasScore, "no [SCORE:] tag means no score, not a random one");
            Assert.AreEqual(0, r.Score);
            Assert.AreEqual("Nice one!", r.Text);
        }

        [Test]
        public void LegacyTags_PraiseWordInProse_DoesNotOverrideAFixStatus()
        {
            // The old keyword scan matched "PERFECT" before "CORRECTION" and mislabelled this.
            var r = ParsedResponse.Parse(
                "[FEEDBACK]\n• Grammar: Almost perfect! One correction: say 'an apple'.\n[/FEEDBACK]\n" +
                "[SPOKEN]Nice try![/SPOKEN]");

            Assert.AreEqual("NEEDS_FIX", r.FeedbackStatus);
        }

        [Test]
        public void LegacyTags_ExplicitStatusTagBeatsKeywords()
        {
            var r = ParsedResponse.Parse(
                "[FEEDBACK]\n[STATUS:PERFECT]\n• Grammar: No correction needed at all.\n[/FEEDBACK]\n" +
                "[SPOKEN]Well done![/SPOKEN]");

            Assert.AreEqual("PERFECT", r.FeedbackStatus);
        }

        [Test]
        public void LegacyTags_UnclosedBlocks_StillYieldSpokenText()
        {
            // A full sentence, so this stays a test of unclosed blocks. With the two-word
            // "hello there" it used to carry, the fragment ceiling rightly holds 90 to 55 - a
            // greeting has no verb, which is the rubric's own test - and the parsing assertion
            // underneath would have failed for a reason that has nothing to do with parsing.
            var r = ParsedResponse.Parse(
                "[TRANSCRIPTION]hello there, how are you today\n[FEEDBACK]\n[SCORE:90]\n• Grammar: Fine.\n[SPOKEN]Hi! How are you?");

            Assert.AreEqual("hello there, how are you today", r.UserTranscription);
            Assert.AreEqual(90, r.Score);
            Assert.AreEqual("Hi! How are you?", r.Text);
        }

        [Test]
        public void QuestTags_AreCollectedAndDeduplicated()
        {
            var r = ParsedResponse.Parse(
                "[QUEST_DONE:2][SPOKEN]Great ordering![/SPOKEN][QUEST_COMPLETE:2][QUEST_DONE:1]");

            CollectionAssert.AreEqual(new[] { 1, 2 }, r.CompletedQuests);
            Assert.AreEqual("Great ordering!", r.Text);
        }

        [Test]
        public void LegacyTags_SubSentenceTurn_EarnsNoQuestCredit()
        {
            // The structured path has always stripped this credit, but the rule was inline in
            // that path only, so a legacy payload kept it. Found by the perturbation corpus.
            var r = ParsedResponse.Parse(
                "[FEEDBACK][SCORE:40][STATUS:NEEDS_FIX]" +
                "• Grammar: Could you tell me the total price, please?[/FEEDBACK]" +
                "[QUEST_DONE:3][SPOKEN]You have not ordered anything yet.[/SPOKEN]");

            Assert.IsTrue(r.HasScore);
            Assert.AreEqual(40, r.Score);
            CollectionAssert.IsEmpty(r.CompletedQuests,
                "a turn below the sentence band cannot clear an objective, on either wire format");
        }

        [Test]
        public void QuestTags_AreStrippedFromSpokenText()
        {
            var r = ParsedResponse.Parse("[SPOKEN]Nice work! [QUEST_DONE:3][/SPOKEN]");

            Assert.AreEqual("Nice work!", r.Text);
            StringAssert.DoesNotContain("QUEST", r.Text);
        }

        // ──────────────────── DEGRADATION ────────────────────

        [Test]
        public void PlainProseWithNoTags_IsTreatedAsDialogue()
        {
            var r = ParsedResponse.Parse("Hello! How can I help you today?");

            Assert.AreEqual("Hello! How can I help you today?", r.Text);
            Assert.IsFalse(r.HasScore);
            Assert.IsFalse(r.HasFeedback);
        }

        [Test]
        public void TagsOnlyWithNoDialogue_ProducesEmptyTextNotRawMarkup()
        {
            // Regression: the avatar used to read its own markup aloud in this case.
            var r = ParsedResponse.Parse("[FEEDBACK]\n[SCORE:88]\n• Grammar: Fine.\n[/FEEDBACK]");

            Assert.AreEqual("", r.Text);
            StringAssert.DoesNotContain("[", r.Text);
        }

        [Test]
        public void NullOrEmptyInput_ReturnsSafeDefaults()
        {
            foreach (string input in new[] { null, "", "   " })
            {
                var r = ParsedResponse.Parse(input);

                Assert.AreEqual("", r.Text);
                Assert.IsFalse(r.HasScore);
                Assert.IsFalse(r.HasFeedback);
                Assert.AreEqual("GOOD", r.FeedbackStatus);
                Assert.AreEqual(AvatarEmotion.Neutral, r.Emotion);
                Assert.IsNotNull(r.CompletedQuests);
            }
        }

        [Test]
        public void MalformedJson_FallsBackToTagParsing()
        {
            // Truncated mid-object, but the tag content is still recoverable.
            var r = ParsedResponse.Parse("{ \"reply\": \"oops [SPOKEN]Recovered text[/SPOKEN]");

            Assert.AreEqual("Recovered text", r.Text);
        }

        // ──────────────────── FRAGMENT CEILING ────────────────────

        private static string Scored(int score, string status = "PERFECT", string transcription = "") => @"{
            ""transcription"": """ + transcription + @""",
            ""reply"": ""Of course."",
            ""score"": " + score + @",
            ""status"": """ + status + @""",
            ""correction"": """",
            ""questsCompleted"": [],
            ""emotion"": ""neutral""
        }";

        [Test]
        public void AOneWordAnswerCannotScorePerfect()
        {
            // Recorded live turn, gemini-3.5-flash-lite, 2026-09-25: a single word scored 100,
            // which plays the 90+ cheer and writes a perfect turn into the learner's history.
            var r = ParsedResponse.Parse(Scored(100), "Painkiller");

            Assert.AreEqual(ParsedResponse.FragmentScoreCeiling, r.Score);
            Assert.AreNotEqual("PERFECT", r.FeedbackStatus, "a capped fragment is not labelled Perfect");
        }

        [Test]
        public void ATwoWordNounPhraseIsHeldToTheCeiling()
        {
            // Same run: "No allergy" scored 88.
            var r = ParsedResponse.Parse(Scored(88, "GOOD"), "No allergy");

            Assert.AreEqual(ParsedResponse.FragmentScoreCeiling, r.Score);
        }

        [Test]
        public void TheRubricsOwnFragmentExamplesAreHeldToTheCeiling()
        {
            foreach (string fragment in new[] { "Coffee please", "Total price", "Yes, please." })
                Assert.AreEqual(ParsedResponse.FragmentScoreCeiling,
                                ParsedResponse.Parse(Scored(95), fragment).Score, fragment);
        }

        [Test]
        public void AContractionMakesTwoWordsASentence()
        {
            // "I'm fine." is two words by this count and a complete sentence. Marking a
            // learner down for it would be a worse fault than the one the ceiling prevents.
            foreach (string sentence in new[] { "I'm fine.", "It's okay.", "That's right.", "I don't." })
                Assert.AreEqual(95, ParsedResponse.Parse(Scored(95), sentence).Score, sentence);
        }

        [Test]
        public void AnImperativeIsASentence()
        {
            foreach (string sentence in new[] { "Sit down.", "Come in.", "Thank you.", "Please wait." })
                Assert.AreEqual(95, ParsedResponse.Parse(Scored(95), sentence).Score, sentence);
        }

        [Test]
        public void AShortSentenceIsNotTouched()
        {
            Assert.AreEqual(96, ParsedResponse.Parse(Scored(96), "Yes, I do.").Score);
        }

        [Test]
        public void AFragmentAlreadyUnderTheCeilingKeepsItsScore()
        {
            // Only ever lowers.
            var r = ParsedResponse.Parse(Scored(40, "NEEDS_FIX"), "Total price");

            Assert.AreEqual(40, r.Score);
            Assert.AreEqual("NEEDS_FIX", r.FeedbackStatus);
        }

        [Test]
        public void AVoiceTurnIsJudgedFromItsTranscription()
        {
            // No client copy of what was said on the voice path; the transcription stands in.
            var r = ParsedResponse.Parse(Scored(100, transcription: "Painkiller"), null);

            Assert.AreEqual(ParsedResponse.FragmentScoreCeiling, r.Score);
        }

        private const string PerfectWithTips = @"{
            ""reply"": ""Of course."",
            ""score"": 100,
            ""status"": ""PERFECT"",
            ""correction"": """",
            ""tip"": ""Add a verb to make it a full sentence."",
            ""betterPhrasing"": ""Do you have a painkiller?"",
            ""questsCompleted"": [],
            ""emotion"": ""happy""
        }";

        [Test]
        public void ACappedFragmentLosesThePerfectConfirmation()
        {
            // The confirmation is composed from the model's label before the ceiling runs, so
            // without this the card read 55 above "Great! Your sentence is grammatically correct."
            var r = ParsedResponse.Parse(PerfectWithTips, "Painkiller");

            Assert.AreEqual(ParsedResponse.FragmentScoreCeiling, r.Score);
            Assert.AreEqual("GOOD", r.FeedbackStatus);
            StringAssert.DoesNotContain("Great!", r.FeedbackText);
            StringAssert.DoesNotContain("Grammar:", r.FeedbackText);
            StringAssert.Contains("Quick Tip: Add a verb to make it a full sentence.", r.FeedbackText);
            StringAssert.Contains("Better Phrasing: Do you have a painkiller?", r.FeedbackText);
            Assert.AreEqual("Do you have a painkiller?", r.BetterPhrasing);
            Assert.IsTrue(r.HasFeedback);
        }

        [Test]
        public void ASentenceKeepsThePerfectConfirmation()
        {
            var r = ParsedResponse.Parse(PerfectWithTips, "Do you have a painkiller?");

            Assert.AreEqual(100, r.Score);
            Assert.AreEqual("PERFECT", r.FeedbackStatus);
            StringAssert.Contains("Grammar: Great! Your sentence is grammatically correct.", r.FeedbackText);
            StringAssert.Contains("Quick Tip:", r.FeedbackText);
            StringAssert.Contains("Better Phrasing:", r.FeedbackText);
        }

        [Test]
        public void WithNothingToJudgeTheScoreStands()
        {
            // No transcription and no utterance: no opinion, not a guess.
            Assert.AreEqual(100, ParsedResponse.Parse(Scored(100), null).Score);
        }
    }
}
