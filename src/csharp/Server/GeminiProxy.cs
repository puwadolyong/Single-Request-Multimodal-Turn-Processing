using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using GeminiAvatar;

namespace Convia.LicenseServer;

/// <summary>
///     Turns an application-level turn into a Gemini request, sends it, and hands the answer
///     back untouched.
///
///     <para>
///     Everything that costs money is decided here and not by the caller: the model, the token
///     ceiling, the rubric and the response schema. A modified client can change what the learner
///     said; it cannot change what the operator pays per turn.
///     </para>
/// </summary>
public sealed class GeminiProxy(HttpClient http, string apiKey, ILogger<GeminiProxy> log)
{
    /// <summary>
    ///     What runs when nothing says otherwise.
    ///
    ///     <para>
    ///     It was gemini-3.5-flash, the model the manuscript's figures were measured on. Google
    ///     deprecated that model in October 2026 and redirects requests for it to
    ///     gemini-3.6-flash - checked: a request naming 3.5 comes back with modelVersion
    ///     "gemini-3.6-flash". Naming 3.6 here says what actually answers, and keeps working
    ///     once the redirect is withdrawn. 3.6 accepts thinkingBudget 0 (zero thought tokens)
    ///     and thinks about 190 tokens per request without it, so the budget is still sent.
    ///     </para>
    /// </summary>
    public const string DefaultModel = "gemini-3.6-flash";

    /// <summary>
    ///     The model every turn goes to. Chosen by the operator, never taken from the request.
    ///
    ///     <para>
    ///     Read from CONVIA_GEMINI_MODEL rather than written here, because the model is the
    ///     largest single lever on what a turn costs - the Flash-Lite tiers are roughly a
    ///     quarter of the price - and changing it should be an environment change on the
    ///     service, reversible in a minute, not a rebuild and a redeploy. A client still
    ///     cannot pick it: the whole point of routing turns through here is that what a turn
    ///     costs is the operator's decision.
    ///     </para>
    /// </summary>
    public static string Model { get; } =
        ResolveModel(Environment.GetEnvironmentVariable("CONVIA_GEMINI_MODEL"));

    /// <summary>
    ///     Turns the configured value into a model name, or the default.
    ///
    ///     <para>
    ///     Checked for shape rather than against a list of known models. A list would need a
    ///     code change for every new release, which is what moving this to configuration was
    ///     for; the shape check is there because the model name goes into the request path,
    ///     and a stray slash or space there sends every turn to the wrong endpoint.
    ///     </para>
    /// </summary>
    internal static string ResolveModel(string? configured)
    {
        string value = (configured ?? "").Trim();
        if (value.Length == 0) return DefaultModel;

        bool looksLikeAModel = value.StartsWith("gemini-", StringComparison.Ordinal)
                               && value.All(c => char.IsAsciiLetterLower(c) || char.IsAsciiDigit(c)
                                                 || c == '-' || c == '.');

        return looksLikeAModel ? value : DefaultModel;
    }

    /// <summary>Whether the configured model was refused and the default used instead.</summary>
    public static bool ModelWasRejected
    {
        get
        {
            string configured = (Environment.GetEnvironmentVariable("CONVIA_GEMINI_MODEL") ?? "").Trim();
            return configured.Length > 0 && configured != Model;
        }
    }

    /// <summary>The answer's token ceiling for an ordinary turn.</summary>
    private const int MaxOutputTokens = 800;

    /// <summary>
    ///     The ceiling for the longest take. The answer repeats what was said - the
    ///     transcription, and often the sentence again in the correction - so a two-minute
    ///     IELTS long turn needs room that a fifteen-second answer never did, and a ceiling
    ///     too low cuts the JSON off mid-string and loses the whole turn. Only tokens actually
    ///     written are billed, so the higher ceiling costs nothing on a short turn.
    /// </summary>
    private const int MaxOutputTokensLongTake = 1600;

    /// <summary>
    ///     The ceiling for this turn: the ordinary one up to thirty seconds of speech, then
    ///     eight tokens for every further second, up to <see cref="MaxOutputTokensLongTake"/>.
    /// </summary>
    internal static int MaxOutputTokensFor(TurnRequest turn)
    {
        double seconds = AudioSeconds(turn);
        if (seconds <= 30) return MaxOutputTokens;
        return Math.Min(MaxOutputTokensLongTake, MaxOutputTokens + (int)Math.Ceiling((seconds - 30) * 8));
    }

    /// <summary>
    ///     Seconds of speech in a voice turn, from the size of its recording. The client sends
    ///     16 kHz mono 16-bit WAV, 32,000 bytes a second, as base64. Zero for a typed turn.
    ///     Used only to size the limits for the turn, never to bill it.
    /// </summary>
    internal static double AudioSeconds(TurnRequest turn)
    {
        int chars = turn?.Audio?.Data?.Length ?? 0;
        if (chars == 0) return 0;
        return Math.Max(0, (chars * 3.0 / 4.0 - 44) / 32000.0);
    }
    private const double Temperature = 0.9;

    /// <summary>Matches GeminiChatService._maxHistoryTurns. The client trims; this enforces.</summary>
    private const int MaxHistoryTurns = 20;

    /// <summary>Enough for any real persona and objective list, far short of a prompt injection payload.</summary>
    public const int MaxScenarioContextChars = 4000;

    /// <summary>
    ///     How many objectives a request may describe.
    ///
    ///     <para>
    ///     Every scenario that ships has three, and the one without objectives has none. Eight
    ///     is room to grow. The cap exists because each rule was truncated while the list was
    ///     not, so a modified client could send ten thousand of them: a single turn carrying
    ///     megabytes of prompt, billed at the input rate, and counted against the quota as one
    ///     turn.
    ///     </para>
    /// </summary>
    public const int MaxObjectives = 10;

    /// <summary>
    ///     How long one utterance may be, the learner's or a stored one.
    ///
    ///     <para>
    ///     The longest take is the IELTS long turn, two minutes, which is around two thousand
    ///     characters of speech - and that transcription comes back into the history. This is
    ///     twice that. It bounds what a modified client can put in a turn and in each of the
    ///     forty messages of history the server forwards - the history was trimmed by count,
    ///     never by size.
    ///     </para>
    /// </summary>
    public const int MaxUtteranceChars = 4000;

    /// <summary>
    ///     The largest recording a turn may carry: 150 s of 16 kHz mono 16-bit, ~6.4 MB as
    ///     base64. The client stops at 120 s for the IELTS long turn and 30 s everywhere
    ///     else, so this is headroom and no more.
    /// </summary>
    public const int MaxAudioBase64Chars = 6_500_000;

    /// <summary>
    ///     How long a turn waits on Gemini before the learner is told it took too long.
    ///
    ///     <para>
    ///     It has to be shorter than the client's own timeout (RequestTimeoutSeconds in
    ///     GeminiChatService, 30 s), so that the server is always the side that gives up. The
    ///     other way round, the client walked away at 25 s while this waited up to 60 s for an
    ///     answer nobody would read - and if the host did not pass the disconnect on, counted
    ///     it against the quota once it came. The learner, meanwhile, was told they could not
    ///     reach the server, which is not what happened.
    ///     </para>
    ///
    ///     <para>
    ///     Measured, not guessed: across five telemetry runs of ~250 text turns each, the
    ///     slowest was 8.6 s and p99 at most 6.6 s. Twenty seconds is more than twice the worst
    ///     seen, and leaves the client ten for a voice upload on a slow connection.
    ///     </para>
    /// </summary>
    public static readonly TimeSpan UpstreamTimeout = TimeSpan.FromSeconds(20);

    /// <summary>The longest any turn waits; the HttpClient's own timeout, as a backstop.</summary>
    public static readonly TimeSpan MaxUpstreamTimeout = TimeSpan.FromSeconds(60);

    /// <summary>
    ///     How long this turn waits on Gemini: <see cref="UpstreamTimeout"/>, plus a quarter
    ///     of a second for every second of speech, since a two-minute recording takes longer
    ///     to listen to than a sentence. Two minutes waits fifty seconds. The client's own
    ///     timeout grows faster (RequestTimeoutFor in GeminiChatService), so the server is
    ///     still always the side that gives up.
    /// </summary>
    public static TimeSpan UpstreamTimeoutFor(TurnRequest turn)
    {
        TimeSpan t = UpstreamTimeout + TimeSpan.FromSeconds(AudioSeconds(turn) / 4.0);
        return t < MaxUpstreamTimeout ? t : MaxUpstreamTimeout;
    }

    public sealed record Result(bool Ok, string Body, string? ErrorCode);

    public async Task<Result> SendTurnAsync(TurnRequest turn, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
            return new Result(false, "", ErrorCodes.NotConfigured);

        string body = BuildRequestJson(turn, Model);

        using var request = new HttpRequestMessage(
            HttpMethod.Post,
            $"https://generativelanguage.googleapis.com/v1beta/models/{Model}:generateContent")
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json")
        };
        request.Headers.Add("x-goog-api-key", apiKey);

        // This turn's own limit, under the HttpClient's longest one.
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(ct);
        limit.CancelAfter(UpstreamTimeoutFor(turn));

        HttpResponseMessage response;
        string payload;
        try
        {
            response = await http.SendAsync(request, limit.Token);

            // Inside the try with the send: the body is read under the same timeout and the
            // same cancellation, and either can fire here as easily as during the send.
            payload = await response.Content.ReadAsStringAsync(limit.Token);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // The client hung up. Nobody is waiting for this answer, so there is nothing to
            // report and nothing to count.
            log.LogInformation("Client left before Gemini answered");
            return new Result(false, "", ErrorCodes.UpstreamError);
        }
        catch (OperationCanceledException e)
        {
            // Not the caller's token, so this turn's limit or HttpClient.Timeout.
            log.LogWarning(e, "Gemini did not answer within {Timeout}", UpstreamTimeoutFor(turn));
            return new Result(false, "", ErrorCodes.UpstreamTimeout);
        }
        catch (Exception e)
        {
            log.LogError(e, "Gemini call failed");
            return new Result(false, "", ErrorCodes.UpstreamError);
        }

        if (!response.IsSuccessStatusCode)
        {
            // The upstream body may name the operator's project or key state, so it is logged
            // and not forwarded.
            log.LogError("Gemini returned {Status}: {Body}", (int)response.StatusCode, payload);
            return new Result(false, "", ErrorCodes.UpstreamError);
        }

        WarnIfThinking(payload);

        // Returned exactly as received. The client's enforcement layer and its six invariants
        // expect Gemini's own shape, and reshaping it here would mean maintaining that contract
        // in two places.
        return new Result(true, payload, null);
    }

    /// <summary>
    ///     Whether this model is sent thinkingBudget 0.
    ///
    ///     <para>
    ///     Measured, not assumed. gemini-3.5-flash thinks unless told not to - 163 thought
    ///     tokens on a one-line grammar question, billed at the output rate - so production
    ///     has always sent a budget of zero, and still does. gemini-3.5-flash-lite refuses that
    ///     same setting with a bare 400 INVALID_ARGUMENT that names no field, which turned
    ///     every turn into an upstream error the moment the model was switched; and it does not
    ///     think by default, measured at zero thought tokens with no setting at all. So the
    ///     field is left out for the Flash-Lite tier rather than replaced with another one.
    ///     </para>
    ///
    ///     <para>
    ///     A rule on the model name is a guess about models not yet measured. That is what
    ///     <see cref="WarnIfThinking"/> is for: if a model this lets through turns out to
    ///     think, the log says so on the first turn rather than the invoice at the end of the
    ///     month.
    ///     </para>
    /// </summary>
    internal static bool SendsThinkingBudget(string model) =>
        !model.Contains("flash-lite", StringComparison.Ordinal);

    /// <summary>
    ///     Logs when a response spent tokens thinking.
    ///
    ///     <para>
    ///     Thought tokens are billed as output but are not in candidatesTokenCount, which is
    ///     the output figure every cost number in this project was worked out from. A model
    ///     that thinks would therefore cost more than every report says it does, with nothing
    ///     anywhere showing why.
    ///     </para>
    /// </summary>
    private void WarnIfThinking(string payload)
    {
        try
        {
            int thoughts = (int?)JObject.Parse(payload)["usageMetadata"]?["thoughtsTokenCount"] ?? 0;
            if (thoughts > 0)
                log.LogWarning("{Model} spent {Thoughts} tokens thinking on a turn. They are billed "
                               + "as output and are not in candidatesTokenCount.", Model, thoughts);
        }
        catch (JsonException)
        {
            // Not this method's business to reject a payload; the client validates it.
        }
    }

    internal static string BuildRequestJson(TurnRequest turn, string model)
    {
        var contents = new JArray();

        // The client already trims, but a client is not a place to enforce anything.
        var history = DropEchoedTurn(turn);
        int skip = Math.Max(0, history.Count - MaxHistoryTurns * 2);
        for (int i = skip; i < history.Count; i++)
        {
            TurnMessage m = history[i];
            contents.Add(new JObject
            {
                ["role"] = m.Role == "model" ? "model" : "user",
                ["parts"] = new JArray
                {
                    new JObject { ["text"] = Truncate(m.Text ?? "", MaxUtteranceChars) }
                }
            });
        }

        var parts = new JArray();
        if (turn.Mode == "voice" && turn.Audio is not null)
        {
            parts.Add(new JObject
            {
                ["text"] = "Listen to my spoken English in this audio. Transcribe it, coach it, "
                         + "reply in character, and list any quest objectives I accomplished. "
                         + "If you cannot hear any actual speech in the audio, return an empty "
                         + "transcription rather than guessing at what I might have said."
            });
            parts.Add(new JObject
            {
                ["inlineData"] = new JObject
                {
                    ["mimeType"] = turn.Audio.MimeType ?? "audio/wav",
                    ["data"] = turn.Audio.Data ?? ""
                }
            });
        }
        else
        {
            parts.Add(new JObject { ["text"] = Truncate(turn.Text ?? "", MaxUtteranceChars) });
        }

        contents.Add(new JObject { ["role"] = "user", ["parts"] = parts });

        var generationConfig = new JObject
        {
            ["temperature"] = Temperature,
            ["maxOutputTokens"] = MaxOutputTokensFor(turn)
        };

        if (SendsThinkingBudget(model))
            generationConfig["thinkingConfig"] = new JObject { ["thinkingBudget"] = 0 };

        generationConfig["responseMimeType"] = CoachTurnSchema.ResponseMimeType;
        generationConfig["responseSchema"] = JObject.Parse(CoachTurnSchema.Json);

        var root = new JObject
        {
            ["systemInstruction"] = new JObject
            {
                ["parts"] = new JArray { new JObject { ["text"] = BuildSystemPrompt(turn) } }
            },
            ["generationConfig"] = generationConfig,
            ["contents"] = contents
        };

        return root.ToString(Formatting.None);
    }

    /// <summary>
    ///     The history, without a copy of the typed line it already ends with.
    ///
    ///     <para>
    ///     Clients built between 2026-09-11 and the fix add the learner's typed line to their
    ///     history before sending, and send it again as the turn's own text, which is appended
    ///     below as the new message. The model therefore received every typed sentence twice
    ///     in a row, treated the repetition as the mistake, and marked perfect sentences down:
    ///     the rubric's own calibration sentence, which it is told scores 98, came back 85 with
    ///     a correction reading "Original: X.X -> Corrected: X".
    ///     </para>
    ///
    ///     <para>
    ///     Fixed here as well as in the client because the client is installed on customers'
    ///     machines, and this fixes every copy already out there the moment it is deployed.
    ///     The condition is narrow on purpose. A correctly built history always ends with the
    ///     model's reply, never with a user message, so a trailing user entry equal to the
    ///     new text can only be the echo - a learner who genuinely repeats themselves has the
    ///     model's reply to their first attempt between the two. Voice turns never had this
    ///     problem and are left alone.
    ///     </para>
    /// </summary>
    internal static List<TurnMessage> DropEchoedTurn(TurnRequest turn)
    {
        var history = turn.History ?? new List<TurnMessage>();
        if (turn.Mode == "voice" || history.Count == 0) return history;

        TurnMessage last = history[^1];
        bool echo = last.Role != "model"
                    && string.Equals((last.Text ?? "").Trim(), (turn.Text ?? "").Trim(),
                                     StringComparison.Ordinal);

        return echo ? history.GetRange(0, history.Count - 1) : history;
    }

    /// <summary>
    ///     Phase 1: the persona and objective rules still travel with the request, because they
    ///     still live in the build. The rubric around them does not, and never did after this
    ///     server existed.
    /// </summary>
    internal static string BuildSystemPrompt(TurnRequest turn)
    {
        string persona = Truncate(turn.ScenarioContext?.Persona ?? "", MaxScenarioContextChars);

        var objectives = new StringBuilder();
        var list = turn.ScenarioContext?.Objectives;
        if (list is { Count: > 0 })
        {
            objectives.Append("SCENARIO QUEST OBJECTIVES TO EVALUATE:\n");
            int count = Math.Min(list.Count, MaxObjectives);
            for (int i = 0; i < count; i++)
                objectives.Append("- Quest ").Append(list[i].Id).Append(": ")
                          .Append(Truncate(list[i].Rule ?? "", 400)).Append('\n');
        }

        // CoachPrompt is the file the Unity client compiles. Same text, one definition.
        return CoachPrompt.Build(persona, objectives.ToString(), turn.ExplainInThai,
                                 string.Equals(turn.Level, "beginner", StringComparison.OrdinalIgnoreCase),
                                 turn.Language ?? "en");
    }

    private static string Truncate(string s, int max) =>
        s.Length <= max ? s : s[..max];
}

public sealed class TurnRequest
{
    public string? Scenario { get; set; }
    public string? Mode { get; set; }
    public string? Text { get; set; }
    public TurnAudio? Audio { get; set; }
    public List<TurnMessage>? History { get; set; }
    public TurnScenarioContext? ScenarioContext { get; set; }
    public bool ExplainInThai { get; set; } = true;

    /// <summary>"beginner" for a learner who chose that level; anything else, or nothing, is the usual level.</summary>
    public string? Level { get; set; }

    /// <summary>
    ///     The language practised: "ja" from the Japanese edition. The English app sends nothing,
    ///     which is English, so clients already installed are unaffected.
    /// </summary>
    public string? Language { get; set; }
}

public sealed class TurnAudio
{
    public string? MimeType { get; set; }
    public string? Data { get; set; }
}

public sealed class TurnMessage
{
    public string? Role { get; set; }
    public string? Text { get; set; }
}

public sealed class TurnScenarioContext
{
    public string? Persona { get; set; }
    public List<TurnObjective>? Objectives { get; set; }
}

public sealed class TurnObjective
{
    public int Id { get; set; }
    public string? Rule { get; set; }
}
