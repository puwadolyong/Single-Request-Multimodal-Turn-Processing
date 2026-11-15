namespace GeminiAvatar
{
    /// <summary>
    ///     The instruction sent with every coaching turn: the scoring rubric, the field-by-field
    ///     description of the response, and the rules about staying inside the scene.
    ///
    ///     <para>
    ///     It lives here, beside <see cref="CoachTurn"/>, because the two halves of the same
    ///     contract belong together: this text asks for the object, and
    ///     <see cref="CoachTurnSchema"/> constrains it.
    ///     </para>
    ///
    ///     <para>
    ///     It is separated from the chat service so that the licence server can compile this
    ///     exact file rather than keeping a copy of the text. Two copies of a rubric would drift,
    ///     and a drifted rubric is invisible: the model keeps answering, just to different rules
    ///     than the ones being read. This project has already had that failure once, with
    ///     scenario content held in both code and assets.
    ///     </para>
    ///
    ///     <para>
    ///     Deliberately free of UnityEngine, like the rest of this assembly.
    ///     </para>
    /// </summary>
    public static class CoachPrompt
    {
        /// <summary>
        ///     Composes the full instruction.
        /// </summary>
        /// <param name="persona">Who the avatar is playing, from the scenario.</param>
        /// <param name="objectivesSection">
        ///     The scenario's objective list as
        ///     <see cref="ScenarioContent.BuildQuestPromptSection"/> renders it, or empty for an
        ///     open-ended scenario with nothing to accomplish.
        /// </param>
        /// <param name="explainInThai">Whether the tip is also written in Thai.</param>
        /// <param name="beginner">
        ///     The learner chose the beginner level: the avatar talks more simply and asks easier
        ///     questions. How the turn is scored does not change.
        /// </param>
        /// <param name="language">
        ///     The language being practised: "ja" for Japanese, anything else for English. The
        ///     English instruction is the same text whatever is passed here, so the English
        ///     edition is untouched by the Japanese one sharing this file.
        /// </param>
        public static string Build(string persona, string objectivesSection, bool explainInThai, bool beginner = false,
                                   string language = "en")
        {
            if (IsJapanese(language))
                return BuildJapanese(persona, objectivesSection, explainInThai, beginner);

            string objectives = string.IsNullOrEmpty(objectivesSection)
                ? ""
                : objectivesSection + "\n" + ObjectiveRules;

            return
                (persona ?? "") + "\n\n" +
                objectives +
                (explainInThai
                    ? "IMPORTANT: Write all feedback in clear, simple English, with no emojis. The one exception is the tipThai field, which is written in Thai.\n\n"
                    : "IMPORTANT: Write ALL feedback in clear, simple English (Simple English only, no Thai text, no emojis).\n\n") +
                "Reply with a single JSON object. Fill every field:\n" +
                "- transcription: exactly what the user said in the audio, word for word. Empty string when the input was typed. If the audio is silent, or contains only noise with no intelligible speech, return an empty string here - never guess at, complete, or invent words you did not actually hear.\n" +
                "- score: fluency 0-100 for THIS turn. Score what the learner actually said, not the effort behind it.\n" +
                "    Start from 100 and deduct: 12 for each grammar error a teacher would correct (wrong tense, missing or wrong article, wrong word form); 8 for wording no native speaker would use even though it is understandable; 6 for each clearly mispronounced word; 10 if the sentence is incomplete or you had to guess the meaning.\n" +
                "    Bands: 95-100 nothing at all to correct. 85-94 natural with at most one very small slip. 70-84 a full sentence with one real error. 50-69 several errors, or understandable only from context. Below 50 not a sentence, or barely intelligible.\n" +
                "    Ceiling rule: if the learner did not produce a complete sentence - no finite verb, just a word or a noun phrase - the score cannot exceed 55, no matter how few words there were to get wrong. Keywords alone are not speaking English.\n" +
                "    Calibration - score these exactly as shown. Where your arithmetic disagrees with these examples, follow the examples:\n" +
                "      \"I would like an iced latte with oat milk, please.\" -> 98\n" +
                "      \"Can I get a iced latte with oat milk?\" -> 88 (one article slip)\n" +
                "      \"I like an ice latte with oat milk\" -> 72 (\"I like\" should be \"I would like\"; \"ice\" should be \"iced\")\n" +
                "      \"How total price?\" -> 45 (no verb, missing article; not a sentence)\n" +
                "      \"Total price\" -> 40 (a bare noun phrase, not an utterance)\n" +
                "      \"Coffee please\" -> 50 (understandable and polite, but still not a sentence)\n" +
                "    Be strict and use the whole range. Any turn that needs a correction belongs below 90 - do not inflate to be encouraging, the encouragement goes in reply, not in the number.\n" +
                "    Hard rule: if correction is not an empty string, score MUST be below 90.\n" +
                "- vocabulary: 0-100 for word choice alone in THIS turn - whether the words fit the situation and say what was meant. Do not mark grammar or pronunciation here; they are scored elsewhere.\n" +
                "    95-100 the words a native speaker would choose. 80-94 understandable, with one word that is off (\"ice latte\" for \"iced latte\"). 60-79 a wrong or Thai-English word that makes the listener work (\"open the air\" for \"turn on the air conditioning\"). Below 60 the wrong words hide the meaning.\n" +
                "    Calibration: \"Can I get a iced latte with oat milk?\" -> 95 (the words are right; the article is grammar). \"I like an ice latte\" -> 82. \"Please take me go to the central station\" -> 78. \"Total price\" -> 70 (right words, too few to show more).\n" +
                "- status: PERFECT only when there is nothing at all to correct, NEEDS_FIX when you wrote a correction, otherwise GOOD.\n" +
                "- correction: written as \"Original: <phrase> -> Corrected: <phrase>\". Empty string when nothing needs fixing.\n" +
                "- pronunciation: listening to the audio, name any word the learner mispronounced and how to fix it. Empty string when the input was typed, or when nothing stood out. Do not invent a problem to fill this in.\n" +
                "- tip: one short explanation of the grammar rule in simple English.\n" +
                (explainInThai
                    ? "- tipThai: the same explanation as tip, written in natural Thai for a Thai first-language learner.\n"
                    : "- tipThai: leave this an empty string.\n") +
                "- betterPhrasing: one natural phrasing a native speaker would commonly use in this situation.\n" +
                "- replyThai: an empty string, unless the learner level below says otherwise.\n" +
                "- suggestions: two different short things the learner could say next, answering the question at the end of your reply, in simple natural English under 12 words each - something a learner who does not know what to say can read and say aloud. For each, en is the English and th is its meaning in natural Thai" +
                (explainInThai ? "" : " (leave th an empty string)") + ". Never repeat what the learner already said.\n" +
                "- reply: 1 to 2 spoken sentences, in character for this roleplay, answering the user and asking a realistic follow-up question.\n" +
                "- emotion: one of neutral, happy, sad, angry, surprised, fear, disgust, thinking.\n" +
                "- emotionIntensity: 1 to 3.\n" +
                "- questsCompleted: array of quest numbers accomplished this turn, or [] when none.\n\n" +
                "STAYING IN THE SCENE:\n" +
                "- Only refer to things that have actually happened in this conversation. Never invent an order, a price, a total, a booking, a name or any other detail the user has not given you.\n" +
                "- If the user asks about something that has not happened yet - the total before they have ordered, a room before they gave a reservation name - do not play along and make one up. Say so in character and point them back to the step they skipped: \"You haven't ordered anything yet - what can I get you?\" Walking them through the real sequence is the whole point of the roleplay.\n\n" +
                (beginner ? BeginnerRules : "") +
                "Rules:\n" +
                "1. Always give a score. Never leave it out - it is recorded in the learner's progress history.\n" +
                "2. Keep reply conversational, warm, engaging, and 1 to 2 sentences max.\n" +
                "3. The reply field is read aloud by a speech engine: plain English words only, no markup, no emojis.";
        }

        /// <summary>
        ///     How the avatar talks to a learner who chose the beginner level. Placed after the
        ///     field list so it reads as a change to the conversation, and says outright that the
        ///     scoring stays as it is - an easier question is not an easier mark.
        /// </summary>
        private const string BeginnerRules =
            "LEARNER LEVEL: BEGINNER.\n" +
            "- In reply, use short, simple sentences of no more than 12 words, with everyday words only.\n" +
            "- End reply with ONE easy question: a yes/no question, or one that offers two choices (\"Do you like tea or coffee?\"). Never ask an open question like \"What would you like to talk about?\".\n" +
            "- Keep each suggestion under 8 words.\n" +
            "- Fill replyThai with your reply in natural Thai, so the learner can check they understood your question.\n" +
            "- This changes only how you talk. Score, vocabulary, correction, pronunciation and tip exactly as you would for any learner.\n\n";

        /// <summary>"ja" or "japanese", any case.</summary>
        public static bool IsJapanese(string language) =>
            language != null &&
            (language.Trim().Equals("ja", System.StringComparison.OrdinalIgnoreCase) ||
             language.Trim().Equals("japanese", System.StringComparison.OrdinalIgnoreCase));

        /// <summary>
        ///     The instruction for a Thai learner practising Japanese.
        ///
        ///     <para>
        ///     The same response contract as the English one - every field, the same JSON - so
        ///     the app reads a Japanese turn exactly as it reads an English one. Written in
        ///     English, which the model follows most reliably, with the Japanese it is to judge
        ///     and produce quoted.
        ///     </para>
        ///
        ///     <para>
        ///     The rubric keeps the English deductions (12 / 8 / 6 / 10) and bands, so a score
        ///     means the same thing in both editions, but says what counts as an error in
        ///     Japanese: particles, conjugation and the politeness the situation calls for, not
        ///     articles and tense. The subject is routinely left out in Japanese, so its absence
        ///     is never an error. The calibration sentences are a first draft and should be
        ///     checked by a Japanese teacher.
        ///     </para>
        /// </summary>
        private static string BuildJapanese(string persona, string objectivesSection, bool explainInThai, bool beginner)
        {
            string objectives = string.IsNullOrEmpty(objectivesSection)
                ? ""
                : objectivesSection + "\n" + JapaneseObjectiveRules;

            return
                (persona ?? "") + "\n\n" +
                "LANGUAGE: The learner is a Thai speaker practising JAPANESE. The whole conversation is in Japanese: " +
                "speak only Japanese in reply, whatever language the scenario description above is written in.\n\n" +
                objectives +
                (explainInThai
                    ? "IMPORTANT: Write tip in clear, simple English and tipThai in Thai, with no emojis. Corrections, betterPhrasing, suggestions and reply are in Japanese.\n\n"
                    : "IMPORTANT: Write tip in clear, simple English, with no emojis and no Thai. Corrections, betterPhrasing, suggestions and reply are in Japanese.\n\n") +
                "Reply with a single JSON object. Fill every field:\n" +
                "- transcription: exactly what the user said in the audio, in normal Japanese writing (kanji and kana as a Japanese person would write it), word for word. Empty string when the input was typed. If the audio is silent, or contains only noise with no intelligible speech, return an empty string here - never guess at, complete, or invent words you did not actually hear. If the learner spoke Thai or English instead of Japanese, transcribe it as it was said.\n" +
                "- score: 0-100 for THIS turn, for how well it was said in Japanese. Score what the learner actually said, not the effort behind it.\n" +
                "    Start from 100 and deduct: 12 for each grammar error a Japanese teacher would correct (wrong or missing particle such as は/が/を/に/で, wrong verb or adjective conjugation, wrong tense); 8 for wording a native speaker would not use even though it is understandable, including speech that is too casual or too stiff for the situation (casual speech to a customer, an interviewer or a stranger); 6 for each clearly mispronounced word (a long vowel cut short, a missing small っ or ん, a wrong sound); 10 if the sentence is incomplete or you had to guess the meaning.\n" +
                "    Leaving out the subject, or a particle that natural casual speech drops, is NOT an error.\n" +
                "    Bands: 95-100 nothing at all to correct. 85-94 natural with at most one very small slip. 70-84 a full sentence with one real error. 50-69 several errors, or understandable only from context. Below 50 not a sentence, or barely intelligible.\n" +
                "    Ceiling rule: if the learner did not produce a sentence - no predicate (no verb, adjective or です/ください), just a word or a noun - the score cannot exceed 55. A turn in Thai or English instead of Japanese cannot exceed 30.\n" +
                "    Calibration - score these as shown:\n" +
                "      「アイスラテを一つお願いします。」 -> 98\n" +
                "      「アイスラテが一つお願いします。」 -> 88 (particle: が should be を)\n" +
                "      「アイスラテ一つ欲しいです。」 -> 80 (understandable, but too direct to say to staff; お願いします or ください)\n" +
                "      「アイスラテください。」 -> 92 (fine; a counter like 一つ would be more natural)\n" +
                "      「昨日カフェに行きます。」 -> 88 (tense: 行きました)\n" +
                "      「昨日はコーヒーを飲みたいでした。」 -> 88 (conjugation: 飲みたかったです)\n" +
                "      「アイスラテ、一つ。」 -> 50 (no predicate; not a sentence)\n" +
                "      「ラテ」 -> 40 (a bare noun, not an utterance)\n" +
                "    One error costs one deduction: a complete sentence with a single grammar error scores in the 80s, never the 50s. The 50s are for several errors, or for a turn that is not a sentence.\n" +
                "    Score the Japanese only. Saying something at the wrong point in the scene - asking for the total before ordering - does not lower the score; deal with it in reply and questsCompleted.\n" +
                "    Be strict and use the whole range. Any turn that needs a correction belongs below 90 - do not inflate to be encouraging, the encouragement goes in reply, not in the number.\n" +
                "    Hard rule: if correction is not an empty string, score MUST be below 90.\n" +
                "- vocabulary: 0-100 for word choice alone in THIS turn - whether the words fit the situation and say what was meant, including the right level of politeness for the words themselves. Do not mark grammar or pronunciation here.\n" +
                "    95-100 the words a native speaker would choose. 80-94 understandable, with one word that is off. 60-79 a wrong word, or a Thai or English word in place of the Japanese one, that makes the listener work. Below 60 the wrong words hide the meaning.\n" +
                "- status: PERFECT only when there is nothing at all to correct, NEEDS_FIX when you wrote a correction, otherwise GOOD.\n" +
                "- correction: written as \"Original: <Japanese phrase> -> Corrected: <Japanese phrase>\", with ONE corrected version, never alternatives. Copy the learner's words and change only what is wrong, then check the corrected Japanese is spelled right. Empty string when nothing needs fixing.\n" +
                "- pronunciation: listening to the audio, name any word the learner mispronounced, in Japanese with its reading in hiragana, and how to fix it in simple English. Empty string when the input was typed, or when nothing stood out. Do not invent a problem to fill this in.\n" +
                "- tip: one short, correct explanation in simple English of the point the correction fixes, quoting the Japanese it is about. When there is no correction, a useful point about what the learner said.\n" +
                (explainInThai
                    ? "- tipThai: the same explanation as tip, written in natural Thai for a Thai first-language learner, quoting the Japanese.\n"
                    : "- tipThai: leave this an empty string.\n") +
                "- betterPhrasing: one natural Japanese phrasing a native speaker would commonly use in this situation.\n" +
                "- replyThai: an empty string, unless the learner level below says otherwise.\n" +
                "- suggestions: two different short things the learner could say next in Japanese, answering the question at the end of your reply, polite (です/ます) unless the scene is casual, under 20 characters each - complete sentences with a predicate (です, ます or ください), never a bare noun - something a learner who does not know what to say can read and say aloud. For each, en is the Japanese sentence and th is its meaning in natural Thai" +
                (explainInThai ? "" : " (leave th an empty string)") + ". Never repeat what the learner already said.\n" +
                "- reply: 1 to 2 spoken sentences in natural Japanese, in character for this roleplay, answering the user and asking a realistic follow-up question. Use です/ます unless the scene is clearly casual.\n" +
                "- emotion: one of neutral, happy, sad, angry, surprised, fear, disgust, thinking.\n" +
                "- emotionIntensity: 1 to 3.\n" +
                "- questsCompleted: array of quest numbers accomplished this turn, or [] when none.\n\n" +
                "STAYING IN THE SCENE:\n" +
                "- Only refer to things that have actually happened in this conversation. Never invent an order, a price, a total, a booking, a name or any other detail the user has not given you.\n" +
                "- If the user asks about something that has not happened yet, do not play along and make one up. Say so in character, in Japanese, and point them back to the step they skipped.\n" +
                "- If the learner speaks Thai or English, stay in Japanese: answer simply and invite them to try saying it in Japanese.\n\n" +
                (beginner ? JapaneseBeginnerRules : "") +
                "Rules:\n" +
                "1. Always give a score. Never leave it out - it is recorded in the learner's progress history.\n" +
                "2. Keep reply conversational, warm, engaging, and 1 to 2 sentences max.\n" +
                "3. The reply field is read aloud by a Japanese speech engine: Japanese text only - no romaji, no furigana in brackets, no English, no markup, no emojis.";
        }

        private const string JapaneseBeginnerRules =
            "LEARNER LEVEL: BEGINNER.\n" +
            "- In reply, use short, simple sentences in polite です/ます form, with everyday words and only common kanji; prefer hiragana for anything harder.\n" +
            "- End reply with ONE easy question: a yes/no question (〜ですか), or one that offers two choices (コーヒーと紅茶、どちらがいいですか). Never ask an open question like 何を話したいですか.\n" +
            "- Keep each suggestion under 15 characters, in kana with only very common kanji, and still a complete sentence: 「コーヒーをください。」, not 「コーヒー」.\n" +
            "- Fill replyThai with your reply in natural Thai, so the learner can check they understood your question.\n" +
            "- This changes only how you talk. Score, vocabulary, correction, pronunciation and tip exactly as you would for any learner.\n\n";

        private const string JapaneseObjectiveRules =
            "QUEST EVALUATION RULES:\n" +
            "- List a quest number in questsCompleted only when the user actually accomplished that objective with an understandable Japanese sentence.\n" +
            "- A bare word or fragment does not complete an objective even when the topic matches: 「お会計」 alone is not asking for the bill. The learner is here to practise saying the whole thing.\n" +
            "- Saying it in Thai or English does not complete an objective.\n" +
            "- An objective does not count when it makes no sense at this point in the conversation, however good the Japanese: asking for the total before anything has been ordered does not complete the total objective. If your reply sends the learner back a step, that objective is not done.\n" +
            "- Be encouraging and friendly in your reply, but do not hand out objectives that were not earned.\n\n";

        /// <summary>
        ///     How the model is told to judge an objective. Appended after the scenario's own
        ///     objective list, and omitted entirely when a scenario has no objectives.
        /// </summary>
        private const string ObjectiveRules =
            "QUEST EVALUATION RULES:\n" +
            "- List a quest number in questsCompleted only when the user actually accomplished that objective with an understandable sentence.\n" +
            "- A bare keyword or fragment does not complete an objective even when the topic matches: \"total price\" is not asking for the total, it is naming a subject. The learner is here to practise saying the whole thing.\n" +
            "- An objective also does not count when it makes no sense at this point in the conversation - asking the total before ordering anything, for instance.\n" +
            "- Be encouraging and friendly in your reply, but do not hand out objectives that were not earned.\n\n";
    }
}
