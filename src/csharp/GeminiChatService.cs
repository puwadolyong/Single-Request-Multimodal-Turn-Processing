using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

namespace GeminiAvatar
{
    /// <summary>
    ///     Multi-turn conversational chat service using Gemini API.
    ///     Supports 8 Scenarios (Free Talk, Cafe, Airport, Interview, Hotel, Shopping, Taxi, Doctor)
    ///     and evaluates scenario quest milestones dynamically.
    ///
    ///     <para>
    ///     Wakes before the other components because it sets <see cref="ScenarioLanguage"/>,
    ///     which the interface reads while it builds its text.
    ///     </para>
    /// </summary>
    [DefaultExecutionOrder(-200)]
    public class GeminiChatService : MonoBehaviour
    {
        [Header("Licence Server")]
        [Tooltip("Base address of the licence server. The Gemini key lives there, not here.")]
        [SerializeField] private string _licenseServerUrl = LicenseCredentials.DefaultServerUrl;

        [Header("Coaching")]
        [Tooltip("Also explain each grammar tip in Thai. Helps Thai first-language learners grasp the rule; costs a few extra output tokens per turn. Requires a Thai-capable TMP font - run Tools > IMEX > Set Up Thai Font Fallback.")]
        [SerializeField] private bool _explainInThai = true;

        [Tooltip("The language being practised: \"ja\" for the Japanese edition, \"en\" for English. Decides the coach's instruction on the server.")]
        [SerializeField] private string _targetLanguage = "en";

        [Header("Scenario Content")]
        [Tooltip("Optional. Scenario text is read from this library when assigned; any scenario it does not define falls back to the built-in defaults. Generate one via Tools > IMEX.")]
        [SerializeField] private PracticeScenarioLibrary _scenarioLibrary;

        [Header("Roleplay Scenario")]
        [SerializeField] private PracticeScenario _currentScenario = PracticeScenario.FreeTalk;

        [Header("Generation Settings")]
        [SerializeField] [Range(1, 50)] private int _maxHistoryTurns = 20;

        /// <summary>
        ///     How long a turn may take, upload included, before it is abandoned.
        ///
        ///     <para>
        ///     Longer than the server's wait on Gemini (GeminiProxy.UpstreamTimeout, 20 s) on
        ///     purpose: the server should always be the one that gives up, and say so with
        ///     upstream_timeout, rather than this giving up first and telling a learner whose
        ///     connection is fine that it cannot reach the server. The ten seconds between the
        ///     two are for uploading a 15 s recording on a slow connection.
        ///     </para>
        /// </summary>
        private const int RequestTimeoutSeconds = 30;

        /// <summary>
        ///     The timeout for one turn: <see cref="RequestTimeoutSeconds"/>, plus three quarters
        ///     of a second for every second of speech. A two-minute IELTS long turn is ~5 MB to
        ///     upload, forty seconds on a slow 1 Mbit/s uplink, and the server waits longer on
        ///     Gemini for it too (GeminiProxy.UpstreamTimeoutFor, a quarter of a second per
        ///     second) - so this grows faster than the server's limit and the server still gives
        ///     up first. The server's tests repeat this rule; change both together.
        /// </summary>
        private static int RequestTimeoutFor(int audioBase64Chars)
        {
            double seconds = audioBase64Chars <= 0 ? 0 : Math.Max(0, (audioBase64Chars * 3.0 / 4.0 - 44) / 32000.0);
            return RequestTimeoutSeconds + (int)Math.Ceiling(seconds * 0.75);
        }

        private GeminiRequest _session;
        private bool _isProcessing;

        /// <summary>Whether a request is currently in-flight.</summary>
        public bool IsProcessing => _isProcessing;

        /// <summary>
        ///     The server's error code for the most recent failed turn, or empty.
        ///
        ///     <para>
        ///     For callers that must react differently to different refusals. The study runner
        ///     is the one that needed it: a rate-limited turn was never spent and only has to be
        ///     sent again a few seconds later, while an exhausted allowance means stop - and the
        ///     learner-facing sentence that arrives in the error callback is not something to
        ///     branch on.
        ///     </para>
        /// </summary>
        public string LastErrorCode { get; private set; } = "";

        /// <summary>Where turns are sent, after any environment override.</summary>
        public string ResolvedTurnUrl => TurnUrl;

        /// <summary>The current conversation turn count.</summary>
        public int TurnCount => _session?.contents?.Count ?? 0;

        /// <summary>The currently active roleplay scenario.</summary>
        public PracticeScenario CurrentScenario => _currentScenario;

        /// <summary>Raised when processing state changes.</summary>
        public event Action<bool> OnProcessingStateChanged;

        /// <summary>
        ///     The allowance the server reported on the last turn, as (used, limit).
        ///
        ///     <para>
        ///     The server is the only thing that knows this. Counting turns locally would
        ///     drift the moment a customer used the licence on a second machine, and would
        ///     quietly disagree with the number that actually decides whether a turn is
        ///     allowed.
        ///     </para>
        /// </summary>
        public event Action<int, int> OnQuotaReported;

        /// <summary>
        ///     Raised when a turn was refused for a reason the licence screen should answer,
        ///     carrying the server's error code.
        ///
        ///     <para>
        ///     Only for refusals that actually stop the learner. A rate limit arrives as the
        ///     same 429 as an exhausted allowance and must not raise this: those turns are not
        ///     spent, the wait is seconds, and throwing up the licence screen would turn a
        ///     pause into what looks like the end of the licence.
        ///     </para>
        /// </summary>
        public event Action<string> OnLicenceRefused;

        /// <summary>
        ///     Codes that mean this installation cannot carry on until something changes.
        ///
        ///     <para>
        ///     Matches the set the licence screen has wording for. Anything else - a bad
        ///     request, an oversized recording, the model being briefly unavailable - is a
        ///     problem with one turn, not with the licence, and is reported in the line under
        ///     the conversation where it belongs.
        ///     </para>
        /// </summary>
        private static bool IsLicenceRefusal(string code)
        {
            switch (code)
            {
                case "invalid_key":
                case "expired":
                case "payment_failed":
                case "revoked":
                case "seat_limit":
                case "quota_exceeded":
                    return true;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     The one call a turn makes. The model is chosen by the server, so the model name
        ///     configured here no longer travels with the request.
        /// </summary>
        private string TurnUrl =>
            LicenseCredentials.ResolveServerUrl(_licenseServerUrl).TrimEnd('/') + "/v1/turn";

        /// <summary>
        ///     Passes on the allowance headers the server attaches to every answer.
        ///
        ///     <para>
        ///     Called before the result is inspected, because a refusal carries them too: a
        ///     turn rejected for being over the limit is exactly when the learner most needs
        ///     the figure to be current.
        ///     </para>
        /// </summary>
        private void ReportQuota(UnityWebRequest request)
        {
            string limit = request.GetResponseHeader("X-Quota-Limit");
            string used = request.GetResponseHeader("X-Quota-Used");

            if (int.TryParse(limit, out int limitTurns) && int.TryParse(used, out int usedTurns))
                OnQuotaReported?.Invoke(usedTurns, limitTurns);
        }

        private void Awake()
        {
            // Before anything reads scenario content: the Japanese edition's scenarios are
            // Japanese.
            ScenarioLanguage.Current = string.IsNullOrWhiteSpace(_targetLanguage) ? "en" : _targetLanguage.Trim();

            // The report and its summaries cannot see the scenarios: tell them the language and,
            // for the Japanese edition, what each scenario is called - "カフェで注文 (Cafe)".
            ReportLocale.Japanese = ScenarioLanguage.IsJapanese;
            ReportLocale.ScenarioName = ScenarioLanguage.IsJapanese ? (Func<string, string>)ReportScenarioName : null;

            InitializeSession();
        }

        /// <summary>
        ///     "[Cafe] カフェで注文" → "カフェで注文 (Cafe)": the scene in the language practised,
        ///     and the English tag a teacher who does not read Japanese can still go by.
        /// </summary>
        private static string ReportScenarioName(string enumName)
        {
            if (!Enum.TryParse(enumName, out PracticeScenario scenario)) return null;

            string label = (BuiltInScenarios.Get(scenario).MenuLabel ?? "").Trim();
            int close = label.IndexOf(']');
            if (label.StartsWith("[") && close > 1 && close < label.Length - 1)
                return label.Substring(close + 1).Trim() + " (" + label.Substring(1, close - 1) + ")";
            return label.Length > 0 ? label : null;
        }

        /// <summary>
        ///     Resets conversation history and starts fresh.
        ///
        ///     <para>
        ///     The session holds the dialogue and nothing else now. The instruction and the
        ///     generation settings it used to carry are the server's to choose, and leaving
        ///     stale copies here would mean reading rules on this side that no longer decide
        ///     anything.
        ///     </para>
        /// </summary>
        public void InitializeSession()
        {
            _session = new GeminiRequest { contents = new List<GeminiContent>() };
        }

        public void SetScenario(PracticeScenario scenario)
        {
            _currentScenario = scenario;
            _openingLine = "";
            InitializeSession();
        }

        /// <summary>
        ///     What the avatar opened this conversation with. The greeting is spoken on the
        ///     client and never sent as a turn, so without this the model did not know it had
        ///     asked "What did you eat for breakfast today?" when the learner answered "Rice".
        /// </summary>
        private string _openingLine = "";

        public void SetOpeningLine(string line) => _openingLine = (line ?? "").Trim();

        /// <summary>
        ///     The learner chose the beginner level: the avatar talks more simply and asks
        ///     easier questions. Sent with every turn; scoring is not affected.
        /// </summary>
        public bool Beginner { get; set; }

        /// <summary>The line the avatar opens this scenario with.</summary>
        public string GetInitialGreeting(PracticeScenario scenario)
        {
            return PracticeScenarioLibrary.ResolveOrBuiltIn(_scenarioLibrary, scenario).Greeting;
        }

        /// <summary>Objectives shown on the quest board for this scenario.</summary>
        public List<ScenarioQuestInfo> GetScenarioQuests(PracticeScenario scenario)
        {
            return PracticeScenarioLibrary.ResolveOrBuiltIn(_scenarioLibrary, scenario).ToQuestInfos();
        }

        /// <summary>
        ///     The scenario's own contribution to the instruction: who the avatar plays and how
        ///     each objective is marked. Sent to the server with every turn, which composes the
        ///     full instruction around it from the shared <see cref="CoachPrompt"/>.
        /// </summary>
        private ScenarioContent CurrentScenarioContent() =>
            PracticeScenarioLibrary.ResolveOrBuiltIn(_scenarioLibrary, _currentScenario);

        /// <summary>
        ///     Sends a user's voice recording directly to Gemini in a single multimodal request.
        /// </summary>
#if UNITY_EDITOR
        /// <summary>
        ///     Each voice turn's request body, as sent: scenario, history and recording. For the
        ///     batch runner, which saves them so the same recording can be replayed to two models -
        ///     speaking the same sentence twice is two different recordings, and a comparison
        ///     made that way measures the speaker as much as the model. Editor only.
        /// </summary>
        public static event Action<string> VoiceRequestBuilt;
#endif

        public void SendVoiceMessage(string base64Wav, Action<ParsedResponse> onSuccess, Action<string> onError)
        {
            StartVoiceTurn(base64Wav, isRetry: false, onSuccess, onError);
        }

        /// <summary>
        ///     Sends a try at saying a corrected sentence again.
        ///
        ///     <para>
        ///     The same request as a voice turn, and billed as one, but outside the
        ///     conversation: no history goes with it, so earlier turns cannot colour what the
        ///     model hears, and nothing is added to history afterwards, so Alice does not carry
        ///     on as if the learner had said the sentence to her. The target sentence is not
        ///     sent either - see <see cref="ProgressRules.JudgeRetry"/> for why.
        ///     </para>
        /// </summary>
        public void SendRetryVoice(string base64Wav, Action<ParsedResponse> onSuccess, Action<string> onError)
        {
            StartVoiceTurn(base64Wav, isRetry: true, onSuccess, onError);
        }

        private void StartVoiceTurn(string base64Wav, bool isRetry, Action<ParsedResponse> onSuccess, Action<string> onError)
        {
            if (_isProcessing)
            {
                onError?.Invoke("Chat is already processing a request. Please wait.");
                return;
            }

            if (string.IsNullOrWhiteSpace(base64Wav))
            {
                onError?.Invoke("Audio recording is empty.");
                return;
            }

            if (!LicenseCredentials.HasKey)
            {
                onError?.Invoke(NoLicenceMessage);
                return;
            }

            StartCoroutine(SendVoiceMessageCoroutine(base64Wav, isRetry, onSuccess, onError));
        }

        private IEnumerator SendVoiceMessageCoroutine(string base64Wav, bool isRetry, Action<ParsedResponse> onSuccess, Action<string> onError)
        {
            // The recording, the conversation so far and the scenario. The instruction the model
            // works to is the server's, composed around these; nothing worded here reaches it.
            if (_session == null)
                InitializeSession();

            TrimHistory();
            string jsonBody = BuildTurnJson(null, base64Wav, includeHistory: !isRetry);

#if UNITY_EDITOR
            VoiceRequestBuilt?.Invoke(jsonBody);
#endif

            return SendTurnCoroutine(jsonBody, isRetry ? "retry" : "voice", null,
                                     base64Wav.Length, !isRetry, onSuccess, onError);
        }

        /// <summary>
        ///     Sends a typed text message to Gemini and returns the assistant's response asynchronously.
        /// </summary>
        public void SendMessage(string userText, Action<ParsedResponse> onSuccess, Action<string> onError)
        {
            if (_isProcessing)
            {
                onError?.Invoke("Chat is already processing a request. Please wait.");
                return;
            }

            if (string.IsNullOrWhiteSpace(userText))
            {
                onError?.Invoke("Message is empty.");
                return;
            }

            if (!LicenseCredentials.HasKey)
            {
                onError?.Invoke(NoLicenceMessage);
                return;
            }

            StartCoroutine(SendMessageCoroutine(userText, onSuccess, onError));
        }

        private IEnumerator SendMessageCoroutine(string userText, Action<ParsedResponse> onSuccess,
            Action<string> onError)
        {
            // The typed line is NOT added to the history before sending. It travels as the
            // request's own "text", and the server appends it as the new message - so adding it
            // to the history first as well sent every typed sentence to the model twice in a
            // row. That is what happened from 2026-09-11, when turns moved behind the licence
            // server, until this was found: the model saw "I would like an iced latte with oat
            // milk, please.I would like an iced latte with oat milk, please.", corrected the
            // repetition, and marked the rubric's own 98 calibration sentence down to 85. Both
            // halves of the exchange are added only once the turn has succeeded.
            if (_session == null)
                InitializeSession();

            TrimHistory();
            return SendTurnCoroutine(BuildTurnJson(userText, null), "text", userText, 0,
                                     true, onSuccess, onError);
        }

        /// <summary>
        ///     Sends one turn and hands back what came of it. Shared by every kind of turn, so a
        ///     fix to how an answer is read or a failure reported reaches all of them.
        ///
        ///     <para>
        ///     A 500, 502 or 503 is tried once more after half a second, typed or spoken: the
        ///     server does not count a turn it could not answer, so a second try costs nothing.
        ///     </para>
        ///
        ///     <para>
        ///     The exchange is added to the history only if the conversation it belongs to is
        ///     still the current one. A scenario change or a cleared history while the request
        ///     was out starts a new conversation, and the old answer must not become the first
        ///     line of it.
        ///     </para>
        /// </summary>
        /// <param name="mode">"text", "voice" or "retry", for the telemetry.</param>
        /// <param name="userText">The typed line, or null for speech, whose words are in the response.</param>
        private IEnumerator SendTurnCoroutine(string jsonBody, string mode, string userText, int audioChars,
                                              bool addToHistory, Action<ParsedResponse> onSuccess,
                                              Action<string> onError)
        {
            _isProcessing = true;
            OnProcessingStateChanged?.Invoke(true);

            if (_session == null)
                InitializeSession();

            TrimHistory();
            GeminiRequest conversation = _session;

            const int maxAttempts = 2;
            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                using (UnityWebRequest request = new UnityWebRequest(TurnUrl, "POST"))
                {
                    request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(jsonBody));
                    request.downloadHandler = new DownloadHandlerBuffer();
                    request.SetRequestHeader("Content-Type", "application/json");
                    ApplyLicenceHeaders(request);
                    request.timeout = RequestTimeoutFor(audioChars);

                    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                    yield return request.SendWebRequest();
                    stopwatch.Stop();
                    ReportQuota(request);

                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        if (IsRetryable(request.responseCode) && attempt < maxAttempts)
                        {
                            Debug.LogWarning($"[Convia] {mode} turn failed [{request.responseCode}] on attempt {attempt}. Retrying in 0.5s...");
                            yield return new WaitForSeconds(0.5f);
                            continue;
                        }

                        FinishTurn(onError, DescribeFailure(request));
                        yield break;
                    }

                    string responseJson = request.downloadHandler.text;
                    GeminiResponse response;
                    try
                    {
                        response = JsonUtility.FromJson<GeminiResponse>(responseJson);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"Failed to parse Gemini response: {ex.Message}\n{responseJson}");
                        FinishTurn(onError, $"Failed to parse response: {ex.Message}");
                        yield break;
                    }

                    if (response == null || response.candidates == null || response.candidates.Count == 0 ||
                        response.candidates[0].content == null ||
                        response.candidates[0].content.parts == null ||
                        response.candidates[0].content.parts.Count == 0)
                    {
                        FinishTurn(onError, "Received empty response from Gemini.");
                        yield break;
                    }

                    string rawText = response.candidates[0].content.parts[0].text;

                    // Recorded before parsing: the study replays the enforcement layer against
                    // exactly what the service returned, so the payload must be untouched here.
                    TurnTelemetry.Record(_currentScenario, mode, userText ?? "", rawText,
                                         response.usageMetadata, stopwatch.Elapsed.TotalMilliseconds,
                                         audioChars, response.modelVersion);

                    // A typed line is the client's own copy of what the learner said, so the
                    // objective-credit rule can be decided without trusting the model's score.
                    // On the speech path recognition happens inside this same request, so the
                    // only transcription in existence is the one the response carries, and
                    // Parse falls back to it.
                    ParsedResponse parsed = ParsedResponse.Parse(rawText, userText);

                    if (addToHistory && ReferenceEquals(conversation, _session))
                    {
                        string said = userText ?? (!string.IsNullOrWhiteSpace(parsed.UserTranscription)
                            ? parsed.UserTranscription
                            : "(Voice message)");

                        // The spoken reply rather than the raw JSON envelope, so the history sent
                        // on every later turn stays small and readable.
                        _session.contents.Add(new GeminiContent("user", said));
                        _session.contents.Add(new GeminiContent("model",
                            string.IsNullOrWhiteSpace(parsed.Text) ? rawText : parsed.Text));
                    }

                    _isProcessing = false;
                    OnProcessingStateChanged?.Invoke(false);
                    onSuccess?.Invoke(parsed);

                    string singleLine = (parsed.Text ?? "").Replace("\n", " ").Replace("\r", "");
                    Debug.Log($"[GeminiChat] {mode}: User=\"{parsed.UserTranscription ?? userText}\" Quests=[{string.Join(",", parsed.CompletedQuests)}] Alice=\"{singleLine}\"");
                    yield break;
                }
            }
        }

        private void FinishTurn(Action<string> onError, string message)
        {
            _isProcessing = false;
            OnProcessingStateChanged?.Invoke(false);
            onError?.Invoke(message);
        }

        /// <summary>
        ///     The turn as the licence server wants it: what the learner said, the conversation
        ///     so far, and which situation they are in.
        ///
        ///     <para>
        ///     Everything that decides what a turn costs - the model, the token ceiling, the
        ///     rubric and the response schema - is chosen by the server and deliberately absent
        ///     here. A modified client can change what the learner said; it cannot change what
        ///     the operator pays for the answer.
        ///     </para>
        ///
        ///     <para>
        ///     The persona and objective rules still travel in scenarioContext because they
        ///     still live in this build. Phase 2 of the server contract moves scenario content
        ///     behind the server and this object disappears.
        ///     </para>
        /// </summary>
        private string BuildTurnJson(string userText, string base64Wav, bool includeHistory = true)
        {
            var history = new JArray();
            foreach (GeminiContent c in includeHistory ? _session.contents : new List<GeminiContent>())
            {
                string text = c.parts != null && c.parts.Count > 0 ? c.parts[0].text : "";
                history.Add(new JObject { ["role"] = c.role, ["text"] = text });
            }

            var root = new JObject
            {
                ["scenario"] = _currentScenario.ToString(),
                ["explainInThai"] = _explainInThai,
                ["level"] = Beginner ? "beginner" : "standard",
                ["language"] = string.IsNullOrWhiteSpace(_targetLanguage) ? "en" : _targetLanguage.Trim(),
                ["history"] = history,
                ["scenarioContext"] = BuildScenarioContext()
            };

            if (base64Wav != null)
            {
                root["mode"] = "voice";
                root["audio"] = new JObject
                {
                    ["mimeType"] = "audio/wav",
                    ["data"] = base64Wav
                };
            }
            else
            {
                root["mode"] = "text";
                root["text"] = userText ?? "";
            }

            return root.ToString(Formatting.None);
        }

        /// <summary>The scenario's persona and marking rules, as authored in this build.</summary>
        private JObject BuildScenarioContext()
        {
            ScenarioContent content = CurrentScenarioContent();

            var objectives = new JArray();
            for (int i = 0; i < content.Quests.Count; i++)
            {
                objectives.Add(new JObject
                {
                    ["id"] = i + 1,
                    ["rule"] = content.Quests[i].EvaluationRule ?? ""
                });
            }

            string persona = content.Persona ?? "";
            if (_openingLine.Length > 0)
                persona += "\n\nYou opened this conversation by saying: \"" + _openingLine + "\" The learner is answering that.";

            return new JObject
            {
                ["persona"] = persona,
                ["objectives"] = objectives
            };
        }

        /// <summary>
        ///     What to tell the learner when a turn fails.
        ///
        ///     <para>
        ///     The server writes a sentence meant to be read by a learner, so that is what is
        ///     shown. What used to be shown was the transport detail with the raw body appended:
        ///     running out of quota mid-conversation put
        ///     <c>API Error [429] ... {"error":{"code":"quota_exceeded" ...}}</c> on screen in
        ///     red, which tells the person practising English nothing they can act on.
        ///     </para>
        ///
        ///     <para>
        ///     The detail is not lost, it moves to the log where it is useful.
        ///     </para>
        /// </summary>
        private string DescribeFailure(UnityWebRequest request)
        {
            string body = request.downloadHandler != null ? request.downloadHandler.text : "";
            Debug.LogError($"[Convia] Turn failed [{request.responseCode}] {request.error}\n{body}");

            // Cleared first, so a failure with no body does not report the code of whatever
            // failed before it.
            LastErrorCode = "";

            // No answer at all is a connection problem, never a licence problem.
            if (request.result == UnityWebRequest.Result.ConnectionError ||
                request.result == UnityWebRequest.Result.DataProcessingError)
            {
                return "Cannot reach the server. Check your internet connection and try again.";
            }

            try
            {
                if (!string.IsNullOrWhiteSpace(body))
                {
                    JToken error = JObject.Parse(body)["error"];

                    string code = (string)error?["code"] ?? "";
                    LastErrorCode = code;
                    if (IsLicenceRefusal(code))
                        OnLicenceRefused?.Invoke(code);

                    JToken message = error?["message"];
                    if (message != null && !string.IsNullOrWhiteSpace(message.ToString()))
                        return message.ToString();
                }
            }
            catch (JsonException)
            {
                // Fall through to the generic wording below.
            }

            return "Something went wrong. Please try again in a moment.";
        }

        /// <summary>
        ///     Whether a failed turn is worth sending again. 502 is included because that is how
        ///     the server reports the model being briefly unavailable, which is exactly the case
        ///     a retry is for; a refused licence or an exhausted quota is not.
        /// </summary>
        private static bool IsRetryable(long status) =>
            status == 500 || status == 502 || status == 503;

        /// <summary>Identifies the licence and the installation on every call.</summary>
        private static void ApplyLicenceHeaders(UnityWebRequest request)
        {
            request.SetRequestHeader("Authorization", "Bearer " + LicenseCredentials.Key);
            request.SetRequestHeader("X-Device-Id", LicenseCredentials.DeviceId);
            request.SetRequestHeader("X-App-Version", Application.version);
        }

        /// <summary>Shown when this installation has no licence key at all.</summary>
        private static string NoLicenceMessage =>
            "No licence key. Set " + LicenseCredentials.EnvironmentVariable
            + " or create " + LicenseCredentials.LocalKeyFilePath + ".";


        /// <summary>Trims conversation history to the configured max turns.</summary>
        private void TrimHistory()
        {
            int maxItems = _maxHistoryTurns * 2; // Each turn = user + model
            while (_session.contents.Count > maxItems)
            {
                _session.contents.RemoveAt(0);
            }
        }

        /// <summary>
        ///     Clears conversation history. A new session rather than emptying the old one, so
        ///     a turn still out when this is called can tell that its conversation has gone.
        /// </summary>
        public void ClearHistory()
        {
            InitializeSession();
        }


        /// <summary>Scenario library in use, or null when running on the built-in defaults.</summary>
        public PracticeScenarioLibrary ScenarioLibrary => _scenarioLibrary;

    }
}
