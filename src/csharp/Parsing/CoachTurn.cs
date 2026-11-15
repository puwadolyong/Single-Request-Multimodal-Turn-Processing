using System.Collections.Generic;
using Newtonsoft.Json;

namespace GeminiAvatar
{
    /// <summary>
    ///     The shape Gemini is asked to return for every coaching turn.
    ///
    ///     This is the contract behind <c>responseSchema</c>: the model fills these fields
    ///     directly instead of embedding <c>[FEEDBACK]</c>-style tags inside prose, so a
    ///     malformed line can no longer make the score or the reply disappear silently.
    /// </summary>
    public class CoachTurn
    {
        /// <summary>Verbatim transcription of the learner's speech. Empty for typed input.</summary>
        [JsonProperty("transcription")]
        public string Transcription { get; set; }

        /// <summary>Fluency score, 0-100. Negative means "the model did not score this turn".</summary>
        [JsonProperty("score")]
        public int Score { get; set; } = -1;

        /// <summary>
        ///     Word choice alone, 0-100: whether the words fit the situation, not whether the
        ///     grammar or the pronunciation was right. Negative when the model gave none - older
        ///     servers never ask for it.
        /// </summary>
        [JsonProperty("vocabulary")]
        public int Vocabulary { get; set; } = -1;

        /// <summary>One of PERFECT, GOOD, NEEDS_FIX.</summary>
        [JsonProperty("status")]
        public string Status { get; set; }

        /// <summary>Corrected sentence, or empty when nothing needed fixing.</summary>
        [JsonProperty("correction")]
        public string Correction { get; set; }

        /// <summary>One short grammar explanation.</summary>
        [JsonProperty("tip")]
        public string Tip { get; set; }

        /// <summary>A more natural way a native speaker would say it.</summary>
        [JsonProperty("betterPhrasing")]
        public string BetterPhrasing { get; set; }

        /// <summary>
        ///     What the learner mispronounced, judged from the audio itself. Empty for typed
        ///     input, and empty when nothing stood out.
        /// </summary>
        [JsonProperty("pronunciation")]
        public string Pronunciation { get; set; }

        /// <summary>
        ///     <see cref="Tip"/> restated in Thai for a Thai first-language learner. Empty when
        ///     Thai explanations are switched off.
        /// </summary>
        [JsonProperty("tipThai")]
        public string TipThai { get; set; }

        /// <summary>The in-character spoken reply. This is what gets sent to TTS.</summary>
        [JsonProperty("reply")]
        public string Reply { get; set; }

        /// <summary>
        ///     <see cref="Reply"/> in Thai, for a beginner checking they understood the question.
        ///     Empty at the usual level, and from older servers.
        /// </summary>
        [JsonProperty("replyThai")]
        public string ReplyThai { get; set; }

        /// <summary>neutral, happy, sad, angry, surprised, fear, disgust or thinking.</summary>
        [JsonProperty("emotion")]
        public string Emotion { get; set; }

        /// <summary>Emotion strength, 1-3.</summary>
        [JsonProperty("emotionIntensity")]
        public int EmotionIntensity { get; set; } = 2;

        /// <summary>
        ///     Two short things the learner could say next, for a learner who does not know what
        ///     to answer. Absent from older servers.
        /// </summary>
        [JsonProperty("suggestions")]
        public List<Suggestion> Suggestions { get; set; }

        /// <summary>Quest objective numbers the learner accomplished this turn.</summary>
        [JsonProperty("questsCompleted")]
        public List<int> QuestsCompleted { get; set; }
    }

    /// <summary>Something the learner could say, in English and in Thai.</summary>
    public class Suggestion
    {
        [JsonProperty("en")]
        public string En { get; set; } = "";

        /// <summary>Its meaning in Thai; empty when Thai explanations are off.</summary>
        [JsonProperty("th")]
        public string Th { get; set; } = "";
    }

    /// <summary>
    ///     The literal <c>responseSchema</c> sent to the Gemini API alongside
    ///     <c>responseMimeType: "application/json"</c>.
    ///
    ///     Kept as a hand-written string rather than generated from <see cref="CoachTurn"/>
    ///     because Gemini accepts only a subset of OpenAPI and rejects the extra keywords a
    ///     generic schema generator emits.
    /// </summary>
    public static class CoachTurnSchema
    {
        /// <summary>Value for generationConfig.responseMimeType.</summary>
        public const string ResponseMimeType = "application/json";

        /// <summary>Value for generationConfig.responseSchema, as a raw JSON object.</summary>
        public const string Json = @"{
  ""type"": ""object"",
  ""properties"": {
    ""transcription"": { ""type"": ""string"" },
    ""score"": { ""type"": ""integer"" },
    ""vocabulary"": { ""type"": ""integer"" },
    ""status"": { ""type"": ""string"", ""enum"": [""PERFECT"", ""GOOD"", ""NEEDS_FIX""] },
    ""correction"": { ""type"": ""string"" },
    ""tip"": { ""type"": ""string"" },
    ""betterPhrasing"": { ""type"": ""string"" },
    ""pronunciation"": { ""type"": ""string"" },
    ""tipThai"": { ""type"": ""string"" },
    ""suggestions"": { ""type"": ""array"", ""items"": { ""type"": ""object"", ""properties"": { ""en"": { ""type"": ""string"" }, ""th"": { ""type"": ""string"" } } } },
    ""reply"": { ""type"": ""string"" },
    ""replyThai"": { ""type"": ""string"" },
    ""emotion"": { ""type"": ""string"", ""enum"": [""neutral"", ""happy"", ""sad"", ""angry"", ""surprised"", ""fear"", ""disgust"", ""thinking""] },
    ""emotionIntensity"": { ""type"": ""integer"" },
    ""questsCompleted"": { ""type"": ""array"", ""items"": { ""type"": ""integer"" } }
  },
  ""required"": [""score"", ""status"", ""reply"", ""emotion""],
  ""propertyOrdering"": [""transcription"", ""score"", ""vocabulary"", ""status"", ""correction"", ""pronunciation"", ""tip"", ""tipThai"", ""betterPhrasing"", ""reply"", ""replyThai"", ""suggestions"", ""emotion"", ""emotionIntensity"", ""questsCompleted""]
}";
    }
}
