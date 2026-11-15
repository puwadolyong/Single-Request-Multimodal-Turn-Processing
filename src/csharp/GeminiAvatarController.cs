using System;
using System.Collections;
using System.Collections.Generic;
using Convai.Domain.DomainEvents.Runtime;
using Convai.Domain.EventSystem;
using Convai.Runtime.Components;
using UnityEngine;

namespace GeminiAvatar
{
    /// <summary>
    ///     Main orchestrator for the Gemini-powered AI avatar.
    ///     Coordinates: User Input → Gemini Chat → Gemini TTS → Audio Playback
    ///     + Convai EventHub injection for animation and emotion.
    /// </summary>
    public class GeminiAvatarController : MonoBehaviour
    {
        [Header("Convai Character (for Animation)")]
        [SerializeField] private ConvaiCharacter _convaiCharacter;

        /// <summary>
        ///     The model to animate, when it is not a Convai character.
        ///
        ///     <para>
        ///     Typed as a GameObject rather than as any particular avatar component so that a
        ///     VRM, an FBX or anything else can be dropped in. Leave it empty and the old
        ///     behaviour stands: the Convai character is found, and failing that whichever
        ///     Animator the scene hands over first.
        ///     </para>
        ///
        ///     <para>
        ///     It exists because that "whichever" is not a choice anybody made.
        ///     FindAnyObjectByType returns an arbitrary one of however many Animators are in the
        ///     scene, so swapping the model meant deleting the old one and hoping - and the two
        ///     components that need to know are built at runtime by this class, so there was no
        ///     field on either of them to set in the Inspector beforehand. This is that field.
        ///     </para>
        /// </summary>
        [Header("Character Model (leave empty to auto-detect)")]
        [Tooltip("The avatar to animate and lip-sync. Set this when using a model that is not " +
                 "a Convai character - a VRM from VRoid, for example. Empty means auto-detect, " +
                 "which picks an arbitrary Animator when the scene holds more than one.")]
        [SerializeField] private GameObject _characterRoot;

        [Header("Scenario Outfits")]
        [SerializeField] private ScenarioWardrobe _wardrobe = new ScenarioWardrobe();

        [Tooltip("Switch off the Convai SDK's own lip-sync component at startup. It drives the " +
                 "same blendshapes as the audio-driven lip sync here, and with both alive the " +
                 "weights alternate from frame to frame and the mouth trembles. Untick only to " +
                 "hand lip sync back to Convai.")]
        [SerializeField] private bool _disableConvaiLipSync = true;

        /// <summary>
        ///     Kokoro speaker ids, as the bundled model (kokoro-en-v0_19) numbers them.
        ///
        ///     <para>
        ///     The values are the model's own, which sort its eleven voices by name with the
        ///     blended default "af" first. They used to follow a ten-voice list that left "af"
        ///     out and swapped two names, so every voice from 1 on was the wrong one - "bf_emma"
        ///     played am_michael, a man. Checked by synthesising each id and measuring its pitch.
        ///     0 is unchanged: it is the voice Alice has always had, and the value scenes store.
        ///     </para>
        /// </summary>
        public enum KokoroVoiceOption
        {
            [InspectorName("af (US Female - Alice default blend)")]
            af_alice = 0,
            [InspectorName("af_bella (US Female - Warm)")]
            af_bella = 1,
            [InspectorName("af_nicole (US Female - Soft)")]
            af_nicole = 2,
            [InspectorName("af_sarah (US Female - Clear)")]
            af_sarah = 3,
            [InspectorName("af_sky (US Female - Bright)")]
            af_sky = 4,
            [InspectorName("am_adam (US Male - Deep)")]
            am_adam = 5,
            [InspectorName("am_michael (US Male - Clear)")]
            am_michael = 6,
            [InspectorName("bf_emma (UK Female)")]
            bf_emma = 7,
            [InspectorName("bf_isabella (UK Female)")]
            bf_isabella = 8,
            [InspectorName("bm_george (UK Male)")]
            bm_george = 9,
            [InspectorName("bm_lewis (UK Male)")]
            bm_lewis = 10
        }

        [Header("Audio")]
        [SerializeField] private AudioSource _audioSource;

        [Header("Services")]
        [SerializeField] private GeminiChatService _chatService;
        [SerializeField] private KokoroTTSService _kokoroService;
        [SerializeField] private SimpleSTTRecorder _sttRecorder;

        [Header("Kokoro-82M Voice Settings (Local Offline TTS)")]
        [Tooltip("Select the Kokoro-82M voice profile.")]
        [SerializeField] private KokoroVoiceOption _kokoroVoice = KokoroVoiceOption.af_alice;

        [Range(0.5f, 2.0f)]
        [Tooltip("Speaking speed multiplier (1.0 = normal, 1.05 = crisp/natural, 1.25 = fast).")]
        [SerializeField] private float _kokoroSpeechSpeed = 1.05f;

        [Range(1, 8)]
        [Tooltip("Number of CPU threads to use for ONNX neural TTS inference (4 is recommended for Ryzen 5 5600).")]
        [SerializeField] private int _kokoroNumThreads = 4;

        [Header("Behavior")]
        [Tooltip("When true, injects events into Convai's EventHub to drive dialogue animation.")]
        [SerializeField] private bool _injectConvaiEvents = true;

        private IEventHub _eventHub;
        private string _characterId;
        private AudioClip _currentClip;
        private Coroutine _audioMonitorCoroutine;
        private bool _isSpeaking;
        private string _currentResponseId;

        /// <summary>Selected Kokoro voice profile.</summary>
        public KokoroVoiceOption KokoroVoice { get => _kokoroVoice; set => _kokoroVoice = value; }

        /// <summary>Kokoro speech speed multiplier.</summary>
        public float KokoroSpeechSpeed { get => _kokoroSpeechSpeed; set => _kokoroSpeechSpeed = Mathf.Clamp(value, 0.5f, 2.0f); }

        /// <summary>
        ///     Changes the voice and how fast it speaks, from the next line on. Handed to the
        ///     voice straight away as well, so a sample played from the settings panel - which
        ///     goes to the voice directly, not through a reply - is already in the new voice.
        /// </summary>
        public void ApplyVoice(KokoroVoiceOption voice, float speed)
        {
            KokoroVoice = voice;
            KokoroSpeechSpeed = speed;

            if (_kokoroService != null)
            {
                _kokoroService.SpeakerId = (int)_kokoroVoice;
                _kokoroService.SpeechSpeed = _kokoroSpeechSpeed;
            }
        }

        /// <summary>Kokoro CPU threads for ONNX inference.</summary>
        public int KokoroNumThreads { get => _kokoroNumThreads; set => _kokoroNumThreads = Mathf.Clamp(value, 1, 8); }

        /// <summary>Local Kokoro-82M TTS service instance.</summary>
        public KokoroTTSService KokoroService => _kokoroService;

        // ──────────────────── STATE ────────────────────

        public enum AvatarState
        {
            Idle,
            Recording,
            Thinking,
            Speaking
        }

        /// <summary>
        ///     The conversation's rules: when the learner may speak, which answers are still
        ///     current, objectives and the victory. Decided there and tested there; carried out
        ///     here.
        /// </summary>
        private readonly ConversationFlow _flow = new ConversationFlow();

        /// <summary>Current avatar state.</summary>
        public AvatarState CurrentState => ToAvatarState(_flow.State);

        /// <summary>Speech-to-text recorder reference for microphone selection and VU meter.</summary>
        public SimpleSTTRecorder STTRecorder => _sttRecorder;

        /// <summary>Raised when state changes.</summary>
        public event Action<AvatarState> OnStateChanged;

        /// <summary>Raised when the character speaks text (for subtitles).</summary>
        public event Action<string> OnSubtitleText;

        /// <summary>Raised when the user's voice input is transcribed into text.</summary>
        public event Action<string> OnUserTranscribedText;

        /// <summary>Raised when an error occurs.</summary>
        public event Action<string> OnError;

        /// <summary>Raised when emotion changes.</summary>
        public event Action<AvatarEmotion, int> OnEmotionChanged;

        /// <summary>Raised when educational grammar feedback is received (feedbackText, status, score).</summary>
        public event Action<string, string, int> OnGrammarFeedback;

        /// <summary>
        ///     Raised once per coached turn with the whole parsed response.
        ///
        ///     <para>
        ///     The progress tracker needs more than the three values <see cref="OnGrammarFeedback"/>
        ///     carries — whether the coach actually rewrote the sentence, for one — so it listens
        ///     here instead of trying to infer that from the score.
        ///     </para>
        /// </summary>
        public event Action<ParsedResponse> OnCoachTurn;

        /// <summary>Raised when the user submits a message (for chat logs/history).</summary>
        public event Action<string> OnUserMessageSent;

        /// <summary>Raised when the practice scenario is changed.</summary>
        public event Action<PracticeScenario> OnScenarioChanged;

        /// <summary>Raised when scenario quest objectives are updated (scenario, questList, completedSet).</summary>
        public event Action<PracticeScenario, List<ScenarioQuestInfo>, HashSet<int>> OnScenarioQuestsUpdated;

        /// <summary>Raised when an individual quest objective is accomplished.</summary>
        public event Action<int> OnQuestCompleted;

        /// <summary>Raised when all 3 quests in the current scenario are accomplished.</summary>
        public event Action OnAllQuestsCompleted;

        /// <summary>Raised when a scenario mission is completed with summary statistics (scenario, avgScore, turns, totalQuests).</summary>
        public event Action<PracticeScenario, int, int, int> OnMissionVictorySummary;

        // ──────────────────── SAYING IT AGAIN ────────────────────

        /// <summary>
        ///     The sentence the latest coached turn asks the learner to say again, or empty when
        ///     it was not worth reviewing. Set before <see cref="OnCoachTurn"/> is raised, so the
        ///     coach card and the progress record offer and keep the same sentence.
        /// </summary>
        public ProgressRules.ReviewTargetResult LastReviewTarget { get; private set; }

        /// <summary>Whether the next spoken take is a try at <see cref="RetryTarget"/>.</summary>
        public bool IsInRetry { get; private set; }

        /// <summary>The sentence being practised while <see cref="IsInRetry"/>.</summary>
        public string RetryTarget { get; private set; } = "";

        private List<string> _retryKeyWords = new List<string>();

        /// <summary>
        ///     Raised when a try at the retry target has been judged: the judgement, what was
        ///     heard, the target, and the review entry the result was recorded on - null when
        ///     the sentence is not in the review list.
        /// </summary>
        public event Action<ProgressRules.RetryJudgement, string, string, MistakeEntry> OnRetryJudged;

        /// <summary>Raised when practice mode starts or ends.</summary>
        public event Action<bool> OnRetryModeChanged;

        /// <summary>What the learner said last, typed or transcribed.</summary>
        private string _lastUtterance = "";

        /// <summary>Currently completed quest IDs (1, 2, 3) in the active scenario.</summary>
        public HashSet<int> CompletedQuests => _flow.CompletedObjectives;

        // ──────────────────── LIFECYCLE ────────────────────

        private bool _started;

        private void Awake()
        {
            _flow.StateChanged += HandleFlowStateChanged;
        }

        private void Start()
        {
            ValidateReferences();
            _started = true;
            _wardrobe?.Apply(_chatService != null ? _chatService.CurrentScenario : PracticeScenario.FreeTalk);
            ResolveConvaiEventHub();
            BroadcastScenarioQuests();
        }

        private void OnDestroy()
        {
            _wardrobe?.Restore();
            // The recorder can outlive this controller (it is added to the same GameObject but
            // found by lookup), so a dangling handler would keep firing into a destroyed object.
            if (_sttRecorder != null)
                _sttRecorder.OnMaxDurationReached -= HandleRecordingLimitReached;

            CleanupAudio();
        }


        /// <summary>
        ///     Points the lip sync and the gestures at whatever character is current.
        ///
        ///     <para>
        ///     Separate from the rest of the setup so it can be run again. Everything else here
        ///     is built once and lasts the session; the character is the one thing that can
        ///     change while the application is running, and re-running only this avoids
        ///     rebuilding the AudioSource and the recorder underneath a live conversation.
        ///     </para>
        /// </summary>
        private void BindCharacter()
        {
            // Ensure Audio-Driven LipSync is attached and bound
            var lipSync = GetComponentInChildren<GeminiAudioLipSync>();
            if (lipSync == null)
            {
                lipSync = gameObject.AddComponent<GeminiAudioLipSync>();
            }
            if (lipSync != null && _audioSource != null)
            {
                // Before InitializeReferences, which is what goes looking when nothing was
                // given to it.
                if (_characterRoot != null)
                    lipSync.SetCharacterRoot(_characterRoot);

                lipSync.SetAudioSource(_audioSource);
                lipSync.InitializeReferences();
            }

            if (_disableConvaiLipSync)
                DisableCompetingLipSync();

            // Ensure Upper Body Gesture Controller is attached and bound
            var gestureCtrl = GetComponentInChildren<GeminiAvatarGestureController>();
            if (gestureCtrl == null)
            {
                gestureCtrl = gameObject.AddComponent<GeminiAvatarGestureController>();
            }
            if (gestureCtrl != null)
            {
                if (_characterRoot != null)
                {
                    var animator = _characterRoot.GetComponentInChildren<Animator>();
                    if (animator != null)
                        gestureCtrl.SetAnimator(animator);
                    else
                        Debug.LogWarning($"[GeminiAvatar] {_characterRoot.name} has no Animator, "
                                         + "so it cannot be posed. Gestures will look elsewhere.");
                }

                gestureCtrl.InitializePlayables();
            }

            // Eyes on the camera while speaking
            var gaze = GetComponentInChildren<GeminiAvatarGaze>();
            if (gaze == null)
                gaze = gameObject.AddComponent<GeminiAvatarGaze>();
            gaze.SetCharacterRoot(_characterRoot);
        }

        /// <summary>
        ///     Changes the character mid-session, re-binding everything that pointed at the old
        ///     one.
        ///
        ///     <para>
        ///     Setting the field alone is not enough once the application is running. The lip
        ///     sync caches a blendshape index per channel and the gesture controller holds a
        ///     PlayableGraph wired to a particular Animator, both built at startup - so a swap
        ///     that only moved the field would leave a new model standing still while the old
        ///     one, now hidden, went on being animated.
        ///     </para>
        ///
        ///     <para>
        ///     Before Start it only records the choice. The binding needs the AudioSource, which
        ///     does not exist yet, and Start will do the work in a moment anyway.
        ///     </para>
        /// </summary>
        public void RebindCharacter(GameObject root)
        {
            _characterRoot = root;

            if (!_started) return;

            Debug.Log($"[GeminiAvatar] Re-binding to {(root != null ? root.name : "nothing")}.");
            BindCharacter();
        }

        private void ValidateReferences()
        {
            if (_chatService == null)
                _chatService = GetComponentInChildren<GeminiChatService>();
            if (_sttRecorder == null)
            {
                _sttRecorder = GetComponentInChildren<SimpleSTTRecorder>();
                if (_sttRecorder == null)
                    _sttRecorder = gameObject.AddComponent<SimpleSTTRecorder>();
            }

            if (_sttRecorder != null)
            {
                _sttRecorder.OnMaxDurationReached -= HandleRecordingLimitReached;
                _sttRecorder.OnMaxDurationReached += HandleRecordingLimitReached;
            }

            // An AudioSource on a deactivated object plays nothing and reports nothing, and the
            // one this scene had assigned lived on the Convai character - so switching that
            // character off to put a different model in its place took Alice's voice with it,
            // with no error anywhere and no obvious connection between the two. The intent below
            // was always to own the AudioSource rather than borrow one; this enforces it.
            if (_audioSource != null && !_audioSource.gameObject.activeInHierarchy)
            {
                Debug.LogWarning(
                    $"[GeminiAvatar] The assigned AudioSource is on '{_audioSource.gameObject.name}', "
                    + "which is not active in the scene, so nothing would be audible. Using a "
                    + "dedicated one instead.");
                _audioSource = null;
            }

            // Create our own dedicated AudioSource so we don't depend on
            // Convai's AudioSource (which gets destroyed during pause/resume)
            if (_audioSource == null)
            {
                var audioGO = new GameObject("GeminiAudio");
                audioGO.transform.SetParent(this.transform);
                _audioSource = audioGO.AddComponent<AudioSource>();
                _audioSource.playOnAwake = false;
                _audioSource.spatialBlend = 0f; // 2D audio
                Debug.Log("[GeminiAvatar] Created dedicated AudioSource for playback.");
            }

            BindCharacter();

            // Ensure Audio Manager (SFX & Ambient BGM) is attached and bound
            var audioMgr = GetComponentInChildren<GeminiAudioManager>();
            if (audioMgr == null)
            {
                audioMgr = gameObject.AddComponent<GeminiAudioManager>();
            }

            if (_chatService == null)
                Debug.LogError("[GeminiAvatar] No GeminiChatService found!");

            if (_kokoroService == null)
                _kokoroService = GetComponentInChildren<KokoroTTSService>();
            if (_kokoroService == null)
                _kokoroService = gameObject.AddComponent<KokoroTTSService>();

            if (_kokoroService != null)
            {
                _kokoroService.SpeakerId = (int)_kokoroVoice;
                _kokoroService.SpeechSpeed = _kokoroSpeechSpeed;
                _kokoroService.NumThreads = _kokoroNumThreads;
            }
        }

        private void OnValidate()
        {
            if (_kokoroService != null)
            {
                _kokoroService.SpeakerId = (int)_kokoroVoice;
                _kokoroService.SpeechSpeed = _kokoroSpeechSpeed;
                _kokoroService.NumThreads = _kokoroNumThreads;
            }
        }

        private void ResolveConvaiEventHub()
        {
            if (!_injectConvaiEvents) return;

            if (_convaiCharacter == null)
            {
                _convaiCharacter = GetComponentInParent<ConvaiCharacter>();
                if (_convaiCharacter == null)
                    _convaiCharacter = FindAnyObjectByType<ConvaiCharacter>();
            }

            if (_convaiCharacter != null)
                _characterId = _convaiCharacter.CharacterId;

            // Try to get the EventHub from ConvaiManager if present
            if (ConvaiManager.ActiveManager != null)
            {
                try
                {
                    _eventHub = ConvaiManager.ActiveManager.Events?.Raw;
                    if (_eventHub != null)
                        Debug.Log($"[GeminiAvatar] Convai EventHub bound. CharacterID={_characterId}");
                }
                catch
                {
                    // Silent fallback - local LipSync handles animation
                }
            }
        }

        /// <summary>
        ///     Switches off any Convai lip-sync component on the character.
        ///
        ///     <para>
        ///     The SDK's character prefab ships with its own lip-sync component enabled, and it
        ///     writes the same jaw and viseme blendshapes that <see cref="GeminiAudioLipSync"/>
        ///     drives. Two writers in one frame means the weights alternate between two values
        ///     and the mouth visibly trembles, intermittently, depending on which component
        ///     happens to run last. Convai's is the one that gives way, because speech here is
        ///     synthesised locally and its component therefore has no viseme stream to work
        ///     from.
        ///     </para>
        ///
        ///     <para>
        ///     Located by type name rather than by a typed reference: the SDK keeps its
        ///     lip-sync module in its own assembly, and a direct reference would make this
        ///     file fail to compile in any project where that module is absent.
        ///     </para>
        /// </summary>
        private void DisableCompetingLipSync()
        {
            GameObject root = _convaiCharacter != null ? _convaiCharacter.gameObject : null;
            if (root == null)
            {
                var found = FindAnyObjectByType<ConvaiCharacter>();
                root = found != null ? found.gameObject : null;
            }

            if (root == null)
                return;

            foreach (var behaviour in root.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (behaviour == null || !behaviour.enabled)
                    continue;

                string typeName = behaviour.GetType().FullName;
                if (string.IsNullOrEmpty(typeName))
                    continue;

                if (typeName.StartsWith("Convai.", StringComparison.Ordinal) &&
                    typeName.IndexOf("LipSync", StringComparison.Ordinal) >= 0)
                {
                    behaviour.enabled = false;
                    Debug.Log($"[GeminiAvatar] Disabled {typeName}: it drives the same " +
                              "blendshapes as the audio-driven lip sync.");
                }
            }
        }

        // ──────────────────── PUBLIC API ────────────────────

        /// <summary>
        ///     Sends a text message to the avatar and triggers the full pipeline:
        ///     Chat → TTS → Audio Playback → Animation.
        /// </summary>
        public void SendTextMessage(string userText)
        {
            if (!_flow.CanSendText)
            {
                Debug.LogWarning($"[GeminiAvatar] Cannot send message in state: {_flow.State}");
                return;
            }

            if (string.IsNullOrWhiteSpace(userText)) return;

            // Practising is spoken; typing a sentence goes back to the conversation.
            if (IsInRetry) CancelRetry();

            _lastUtterance = userText;
            OnUserMessageSent?.Invoke(userText);

            int conversation = _flow.TurnSent();
            _currentResponseId = Guid.NewGuid().ToString("N").Substring(0, 8);

            _chatService.SendMessage(userText,
                onSuccess: response => { if (StillCurrent(conversation)) OnChatResponse(response); },
                onError: error => { if (StillCurrent(conversation)) OnChatError(error); }
            );
        }

        /// <summary>Starts microphone recording for voice input.</summary>
        public void StartVoiceInput()
        {
            if (!_flow.CanStartRecording || _sttRecorder == null) return;

            // Thirty seconds, or two minutes for the IELTS long turn.
            _sttRecorder.MaxRecordingSeconds = CurrentContent.MaxSpeakingSeconds;
            _sttRecorder.StartRecording();

            // No device, or the device refused: there is no take to stop, and a Recording state
            // with nothing behind it ends in a misleading message when the key is released.
            if (!_sttRecorder.IsRecording)
            {
                OnError?.Invoke("The microphone didn't start - check it under SETTINGS > Mic.");
                return;
            }

            _flow.RecordingStarted();
        }

        /// <summary>Stops recording and processes voice input.</summary>
        public void StopVoiceInput()
        {
            if (!_flow.CanStopRecording || _sttRecorder == null) return;

            _sttRecorder.StopRecording((base64Wav, audioClip) =>
            {
                if (string.IsNullOrEmpty(base64Wav))
                {
                    // Idle first, then the message: SetState drives the UI back to its "Ready to
                    // Chat" status line, so raising the error before it wiped the error off the
                    // screen in the same frame and the dropped turn looked like nothing at all
                    // had happened.
                    _flow.TurnEnded();

                    string reason = _sttRecorder != null && !string.IsNullOrEmpty(_sttRecorder.LastDiscardReason)
                        ? _sttRecorder.LastDiscardReason
                        : "I didn't catch any audio - check your microphone and try again.";
                    OnError?.Invoke(reason);
                    return;
                }

                // Send the audio itself to Gemini: it transcribes, coaches pronunciation and
                // replies in one request.
                if (IsInRetry)
                    ProcessRetryVoice(base64Wav);
                else
                    ProcessVoiceDirectly(base64Wav, _sttRecorder != null ? _sttRecorder.LastTakeTiming : default);
            });
        }

        /// <summary>
        ///     Makes the next spoken take a try at <paramref name="target"/> instead of a turn in
        ///     the conversation. Stays on until <see cref="CancelRetry"/>, so a failed try can be
        ///     had again straight away.
        /// </summary>
        /// <param name="keyWords">Words the try must include; see <see cref="ProgressRules.JudgeRetry"/>.</param>
        /// <param name="resumeConversationOnPass">
        ///     True when the try is a correction practised in the middle of the conversation: a
        ///     pass then repeats the avatar's last question and ends the practice, so the next
        ///     thing said answers it. False for the review list, which is not part of any
        ///     conversation and moves on to its next sentence instead.
        /// </param>
        public void BeginRetry(string target, IList<string> keyWords, bool resumeConversationOnPass = false)
        {
            if (string.IsNullOrWhiteSpace(target)) return;

            RetryTarget = target.Trim();
            _retryKeyWords = keyWords != null ? new List<string>(keyWords) : new List<string>();
            _resumeConversationOnPass = resumeConversationOnPass;

            bool changed = !IsInRetry;
            IsInRetry = true;
            if (changed) OnRetryModeChanged?.Invoke(true);
        }

        /// <summary>Back to the conversation.</summary>
        public void CancelRetry()
        {
            if (!IsInRetry) return;

            IsInRetry = false;
            RetryTarget = "";
            _retryKeyWords.Clear();
            OnRetryModeChanged?.Invoke(false);
        }

        /// <summary>Interrupts the current speech.</summary>
        public void InterruptSpeech()
        {
            if (!_flow.CanInterrupt) return;

            // So a piece of this reply still being synthesised is dropped, not queued.
            _speechGeneration++;
            StopAudioPlayback();
            InjectSpeechStopped();

            // Cutting the final reply short does not undo clearing the scenario.
            CelebrateWhenDue(_flow.SpeechInterrupted());
        }

        /// <summary>
        ///     The recorder hit its time limit. Close the take through the normal path so the
        ///     captured audio is still sent, rather than letting the buffer run out.
        /// </summary>
        private void HandleRecordingLimitReached()
        {
            if (!_flow.CanStopRecording) return;
            StopVoiceInput();
        }

        /// <summary>Resets conversation history.</summary>
        public void ResetConversation()
        {
            LeaveConversation(forgetVictory: false);
            _chatService?.ClearHistory();
            _lastReplyText = "";
        }

        /// <summary>
        ///     Ends whatever the current conversation has under way, for a scenario change or a
        ///     cleared history: practice mode, a take being recorded, a turn waiting on the
        ///     server, and speech.
        ///
        ///     <para>
        ///     Each used to survive the switch. A reply still in flight arrived in the new
        ///     scenario - spoken over its greeting, its objectives credited against the new
        ///     ones - and a take being recorded lost its state, so releasing the key no longer
        ///     stopped the microphone.
        ///     </para>
        /// </summary>
        /// <param name="forgetVictory">See <see cref="ConversationFlow.Leave"/>.</param>
        private void LeaveConversation(bool forgetVictory)
        {
            CancelRetry();
            LeaveResult left = _flow.Leave(forgetVictory);

            if (left.CancelRecording && _sttRecorder != null)
                _sttRecorder.CancelRecording();

            // A reply being spoken, or still waiting for the voice, is part of what is being
            // left too.
            _speechGeneration++;
            if (left.StopSpeech)
            {
                StopAudioPlayback();
                InjectSpeechStopped();
            }
            if (_speakCoroutine != null)
            {
                StopCoroutine(_speakCoroutine);
                _speakCoroutine = null;
            }

            CelebrateWhenDue(left.Celebrate);
        }

        /// <summary>Whether a turn sent in conversation <paramref name="sentIn"/> still belongs to the current one.</summary>
        private bool StillCurrent(int sentIn)
        {
            if (_flow.IsCurrent(sentIn)) return true;
            Debug.Log("[GeminiAvatar] Dropping the answer to a turn from a conversation that has ended.");
            return false;
        }

        private readonly System.Random _starterRandom = new System.Random();
        private int _lastFreeTalkTopic = -1;

        /// <summary>
        ///     Starts the scenario already selected with its opening line, as choosing it from the
        ///     menu does. For app start, so Alice speaks first rather than waiting in silence. Does
        ///     nothing once the learner has begun.
        /// </summary>
        public void OpenConversation()
        {
            if (_chatService == null || !_flow.CanStartRecording || _chatService.TurnCount > 0) return;
            ChangeScenario(_chatService.CurrentScenario);
        }

        /// <summary>The learner's level, for how Alice talks; see <see cref="GeminiChatService.Beginner"/>.</summary>
        public void SetLearnerLevel(bool beginner)
        {
            if (_chatService != null) _chatService.Beginner = beginner;
        }

        /// <summary>Switches the roleplay practice scenario and plays the initial opening line.</summary>
        public void ChangeScenario(PracticeScenario scenario)
        {
            // The sentence being practised belongs to the conversation being left, and the
            // card showing it is hidden: the first thing said here is an answer, not a try. A
            // victory still waiting belongs to the scenario being left as well.
            LeaveConversation(forgetVictory: true);
            _wardrobe?.Apply(scenario);
            _flow.StartScenario();

            if (_chatService != null)
            {
                _chatService.SetScenario(scenario);
                string greeting = _chatService.GetInitialGreeting(scenario);

                // Something to say back, from the first line on. Free Talk opens on a topic
                // Alice picks rather than "what would you like to talk about?", which a learner
                // short of English could not answer; a scenario offers its objectives' examples.
                var ideas = new List<Suggestion>();
                string greetingThai = "";
                if (scenario == PracticeScenario.FreeTalk)
                {
                    ConversationStarter starter = ConversationStarters.NextFreeTalk(_starterRandom, ref _lastFreeTalkTopic);
                    greeting = starter.Opener;
                    greetingThai = starter.OpenerTh;
                    foreach (var a in starter.Answers) ideas.Add(new Suggestion { En = a.En, Th = a.Th });
                }
                else
                {
                    foreach (var a in ConversationStarters.ScenarioAnswers(CurrentContent))
                        ideas.Add(new Suggestion { En = a.En, Th = a.Th });
                }
                _chatService.SetOpeningLine(greeting);

                OnScenarioChanged?.Invoke(scenario);
                BroadcastScenarioQuests();

                // Play opening line for the scenario
                // Gated on the greeting alone: the playback path below picks Kokoro first, so
                // requiring the cloud TTS component here left a Kokoro-only rig with no opening line.
                if (!string.IsNullOrWhiteSpace(greeting))
                {
                    var greetingResponse = new ParsedResponse
                    {
                        Text = greeting,
                        FeedbackText = "",
                        FeedbackStatus = "GOOD",
                        HasFeedback = false,
                        Emotion = AvatarEmotion.Happy,
                        EmotionIntensity = 2,
                        Suggestions = ideas,
                        ReplyThai = greetingThai
                    };

                    OnChatResponse(greetingResponse);
                }
            }
        }

        /// <summary>Advances automatically to the next sequential roleplay scenario.</summary>
        /// <summary>
        ///     Advances to the next scenario. Order comes from the assigned scenario library, so
        ///     adding a scenario no longer means editing a switch here.
        /// </summary>
        public void AdvanceToNextScenario()
        {
            if (_chatService == null) return;
            ChangeScenario(PracticeScenarioLibrary.NextAfterOrBuiltIn(
                _chatService.ScenarioLibrary, _chatService.CurrentScenario));
        }

        /// <summary>Scenario library in use, or null when running on the built-in defaults.</summary>
        public PracticeScenarioLibrary ScenarioLibrary => _chatService != null ? _chatService.ScenarioLibrary : null;

        /// <summary>
        ///     Display name for a scenario, resolved through the scenario library when one is
        ///     assigned. Lets the UI show authored names without holding a library reference itself.
        /// </summary>
        public string GetScenarioDisplayName(PracticeScenario scenario)
        {
            return PracticeScenarioLibrary.ResolveOrBuiltIn(
                _chatService != null ? _chatService.ScenarioLibrary : null, scenario).DisplayName;
        }

        /// <summary>
        ///     Objectives the current scenario defines. Zero for open-ended practice.
        ///     Everything about quest progress is measured against this rather than a
        ///     hardcoded three, so Free Talk cannot be "cleared" and an authored scenario
        ///     with two or four objectives works.
        /// </summary>
        private int CurrentObjectiveCount => _chatService == null ? 0 : CurrentContent.Quests.Count;

        /// <summary>The scenario being practised, as authored or built in.</summary>
        private ScenarioContent CurrentContent =>
            PracticeScenarioLibrary.ResolveOrBuiltIn(
                _chatService != null ? _chatService.ScenarioLibrary : null,
                _chatService != null ? _chatService.CurrentScenario : PracticeScenario.FreeTalk);

        public void BroadcastScenarioQuests()
        {
            if (_chatService == null) return;
            var quests = _chatService.GetScenarioQuests(_chatService.CurrentScenario);
            OnScenarioQuestsUpdated?.Invoke(_chatService.CurrentScenario, quests, _flow.CompletedObjectives);
        }

        // ──────────────────── CHAT PIPELINE ────────────────────

        private void OnChatResponse(ParsedResponse response)
        {
            string singleLineText = response.Text.Replace("\n", " ").Replace("\r", "");
            Debug.Log($"[GeminiAvatar] Chat response: [{response.Emotion}:{response.EmotionIntensity}] {singleLineText}");

            // Grammar as the share of the turn the coach left alone. Before any listener hears
            // about the turn, so the card and the record see the same number.
            if (response.HasScore)
                response.GrammarScore = TurnSkills.GrammarAccuracy(_lastUtterance, response.Correction);

            // Worked out once, here, so the card offers the sentence the record keeps.
            bool worthReviewing = response.HasScore && ProgressRules.IsWorthReviewing(
                response.HasCorrection, response.HasPronunciationNote, response.FeedbackStatus,
                response.HasScore, response.Score, ProgressRules.ReviewScoreThreshold);
            LastReviewTarget = worthReviewing
                ? ProgressRules.ReviewTarget(_lastUtterance, response.Correction, response.BetterPhrasing)
                : new ProgressRules.ReviewTargetResult { Sentence = "", KeyWords = new List<string>() };

            // 1. Notify emotion immediately
            OnEmotionChanged?.Invoke(response.Emotion, response.EmotionIntensity);
            InjectEmotion(response.Emotion, response.EmotionIntensity);

            // 2. Notify Grammar Feedback immediately for instant UI display
            if (response.HasFeedback)
            {
                OnGrammarFeedback?.Invoke(response.FeedbackText, response.FeedbackStatus, response.Score);
            }

            OnCoachTurn?.Invoke(response);

            // 3. The turn against the scenario: counted (the greeting is not), objectives
            //    credited, and the victory armed when the last one is done. The rules are in
            //    ConversationFlow.RecordTurn.
            TurnResult turn = _flow.RecordTurn(
                response.HasScore, response.Score, !string.IsNullOrWhiteSpace(response.UserTranscription),
                response.CompletedQuests, CurrentObjectiveCount,
                _chatService != null ? _chatService.CurrentScenario : PracticeScenario.FreeTalk);

            if (turn.NewObjectives.Count > 0)
            {
                foreach (int q in turn.NewObjectives)
                    OnQuestCompleted?.Invoke(q);

                GetComponentInChildren<GeminiAudioManager>()?.PlayQuestSuccessSound();
            }

            // Keep Quest Board UI permanently synchronized with cumulative completed quests
            BroadcastScenarioQuests();

            string speechText = !string.IsNullOrWhiteSpace(response.Text) ? response.Text : "Hello!";
            _lastReplyText = response.Text ?? "";

            // 5. Speak the reply. Kokoro runs locally from StreamingAssets: nothing here is
            //    billed, and nothing leaves the machine.
            Speak(speechText);
        }

        /// <summary>Speaks a line, replacing whatever is being spoken or synthesised.</summary>
        private void Speak(string speechText)
        {
            // Busy until the line plays or is shown instead; see ConversationFlow.SpeechQueued.
            _flow.SpeechQueued();

            _speechGeneration++;
            if (_speakCoroutine != null) StopCoroutine(_speakCoroutine);
            _speakCoroutine = StartCoroutine(SpeakWhenReady(speechText, _speechGeneration));
        }

        /// <summary>The avatar's last conversation reply, for repeating its question after a practice.</summary>
        private string _lastReplyText = "";

        /// <summary>See the parameter of the same name on <see cref="BeginRetry"/>.</summary>
        private bool _resumeConversationOnPass;

        /// <summary>Counts replies to speak; a reply whose number is no longer current is dropped.</summary>
        private int _speechGeneration;
        private Coroutine _speakCoroutine;

        /// <summary>
        ///     How long a reply waits for the local voice before being shown as text instead.
        ///     Long enough for the model to finish loading on a slow machine.
        /// </summary>
        private const float VoiceWaitSeconds = 30f;

        /// <summary>
        ///     Speaks a reply once the local voice can take it.
        ///
        ///     <para>
        ///     It used to be asked straight away and to give up at the first refusal. Kokoro
        ///     synthesises one sentence at a time and spends several seconds loading its model
        ///     after launch, so choosing a scenario soon after starting - or while the previous
        ///     scenario's greeting was still being synthesised - put "Voice playback is
        ///     unavailable" on screen on a machine whose voice was fine, and the greeting that had
        ///     been displaced was then spoken over the new scenario when it finished. Now a reply
        ///     waits its turn, and one overtaken by a newer reply stops quietly: its audio is
        ///     never played.
        ///     </para>
        /// </summary>
        private IEnumerator SpeakWhenReady(string speechText, int generation)
        {
            float deadline = Time.unscaledTime + VoiceWaitSeconds;
            while (_kokoroService != null
                   && (_kokoroService.IsLoading || _kokoroService.IsBusy)
                   && Time.unscaledTime < deadline)
            {
                yield return null;
            }

            _speakCoroutine = null;
            if (generation != _speechGeneration) yield break;

            if (_kokoroService == null || !_kokoroService.IsInitialized)
            {
                ContinueWithoutSpeech(
                    speechText,
                    _kokoroService == null ? "no local voice is installed" : "the local voice is not ready");
                yield break;
            }

            _kokoroService.SpeakerId = (int)_kokoroVoice;
            _kokoroService.SpeechSpeed = _kokoroSpeechSpeed;
            _kokoroService.NumThreads = _kokoroNumThreads;

            List<string> chunks = SpeechChunker.Split(speechText);
            if (chunks.Count == 0) chunks.Add(speechText);
            SynthesiseChunk(chunks, 0, speechText, generation);
        }

        /// <summary>
        ///     Synthesises one piece of a reply and, once it is made, the piece after it.
        ///
        ///     <para>
        ///     The first piece starts the reply playing; later ones join the queue that
        ///     <see cref="MonitorAudioPlayback"/> plays from. One piece at a time, because Kokoro
        ///     takes one request at a time, and the next is asked for from the previous one's
        ///     callback, which runs after Kokoro has marked itself free.
        ///     </para>
        /// </summary>
        private void SynthesiseChunk(List<string> chunks, int index, string speechText, int generation)
        {
            _kokoroService.GenerateSpeech(chunks[index],
                onSuccess: clip =>
                {
                    if (generation != _speechGeneration)
                    {
                        Destroy(clip); // overtaken or interrupted while being synthesised
                        return;
                    }

                    if (index == 0)
                    {
                        OnTTSReady(clip, speechText, chunks.Count - 1);
                    }
                    else
                    {
                        _readyChunks.Enqueue(clip);
                        _chunksToCome--;
                    }

                    if (index + 1 < chunks.Count)
                        SynthesiseChunk(chunks, index + 1, speechText, generation);
                },
                onError: err =>
                {
                    if (generation != _speechGeneration) return;

                    if (index == 0)
                    {
                        ContinueWithoutSpeech(speechText, err);
                    }
                    else
                    {
                        // The whole reply is already on screen as the subtitle, so the turn
                        // ends after what has been spoken rather than failing.
                        Debug.LogWarning($"[GeminiAvatar] Could not voice the rest of the reply ({err}).");
                        _chunksToCome = 0;
                    }
                });
        }

        /// <summary>Pieces of the current reply that are synthesised and waiting to be played.</summary>
        private readonly Queue<AudioClip> _readyChunks = new Queue<AudioClip>();

        /// <summary>Pieces of the current reply still being synthesised.</summary>
        private int _chunksToCome;

        /// <summary>
        ///     Finishes a turn for which no audio could be produced.
        ///
        ///     <para>
        ///     Cloud synthesis used to be tried here when the local voice failed. That meant a
        ///     machine where the Kokoro model did not load quietly began calling a paid service
        ///     with a key shipped inside the build - the one arrangement this application is
        ///     moving away from. The call is gone.
        ///     </para>
        ///
        ///     <para>
        ///     What replaces it is not silence. The reply was previously published only from
        ///     <see cref="OnTTSReady"/>, so a synthesis failure lost the avatar's line as well as
        ///     its voice and the learner was left with nothing at all. The line is published here
        ///     too, and the turn degrades to reading rather than stopping.
        ///     </para>
        /// </summary>
        private void ContinueWithoutSpeech(string speechText, string reason)
        {
            Debug.LogWarning($"[GeminiAvatar] No speech this turn ({reason}); showing the reply as text.");

            OnSubtitleText?.Invoke(speechText);
            OnError?.Invoke("Voice playback is unavailable - Alice's reply is on screen.");

            // No audio means MonitorAudioPlayback never runs, so the line ends here - and with
            // it, a cleared scenario's victory.
            CelebrateWhenDue(_flow.SpeechFinished());
        }

        private void OnChatError(string error)
        {
            Debug.LogError($"[GeminiAvatar] Chat error: {error}");
            OnError?.Invoke(error);
            _flow.TurnEnded();
        }

        // ──────────────────── TTS PIPELINE ────────────────────

        private void OnTTSReady(AudioClip clip, string speechText, int chunksToCome)
        {
            _flow.SpeechStarted();

            // Clean up previous clip
            CleanupAudio();
            _currentClip = clip;
            _chunksToCome = chunksToCome;

            // Play audio
            _audioSource.clip = clip;
            _audioSource.Play();

            // Display subtitle at the exact moment audio playback starts
            OnSubtitleText?.Invoke(speechText);

            // Inject Convai events
            InjectSpeechStarted();
            InjectAudioPlaybackStarted();
            InjectTtsText(speechText);

            // Monitor audio completion
            _audioMonitorCoroutine = StartCoroutine(MonitorAudioPlayback());
        }

        // ──────────────────── VOICE INPUT PIPELINE ────────────────────

        private void ProcessVoiceDirectly(string base64Wav, SpeechTiming timing)
        {
            int conversation = _flow.TurnSent();
            _currentResponseId = Guid.NewGuid().ToString("N").Substring(0, 8);

            // Send voice directly to Gemini in 1 request (combines STT + Grammar Analysis + Response)
            _chatService.SendVoiceMessage(base64Wav,
                onSuccess: response =>
                {
                    if (!StillCurrent(conversation)) return;

                    // Backstop for noise that got past the recorder's silence gate: if the model
                    // heard no words, drop the turn rather than substituting a placeholder and
                    // recording a score against speech that never happened.
                    if (string.IsNullOrWhiteSpace(response.UserTranscription))
                    {
                        Debug.LogWarning("[GeminiAvatar] The model returned no transcription; discarding the turn.");
                        _flow.TurnEnded();
                        OnError?.Invoke("I couldn't make out any speech there - try again a bit closer to the mic.");
                        return;
                    }

                    string transcribedText = response.UserTranscription;
                    _lastUtterance = transcribedText;

                    // How it was said, from the recording: pace and pauses.
                    response.FluencyScore = TurnSkills.Fluency(TurnSkills.CountWords(transcribedText), timing, out int wpm);
                    response.WordsPerMinute = wpm;
                    if (timing.HasSpeech)
                    {
                        response.SpeechSeconds = timing.SpanSeconds;
                        response.PauseSeconds = timing.PauseSeconds;
                    }

                    OnUserTranscribedText?.Invoke(transcribedText);
                    OnUserMessageSent?.Invoke(transcribedText);

                    // Process dialogue response (feedback card, emotion, and TTS)
                    OnChatResponse(response);
                },
                onError: error => { if (StillCurrent(conversation)) OnChatError(error); }
            );
        }

        /// <summary>
        ///     A try at the retry target. Judged here against what was heard, and kept out of
        ///     everything a conversation turn feeds: no reply, no objectives, no score in the
        ///     session average, no chat history, no new review entry. The progress record hears
        ///     about it only through <see cref="LearningProgressService.RecordReview"/>.
        /// </summary>
        private void ProcessRetryVoice(string base64Wav)
        {
            int conversation = _flow.TurnSent();
            string target = RetryTarget;
            var keyWords = new List<string>(_retryKeyWords);

            _chatService.SendRetryVoice(base64Wav,
                onSuccess: response =>
                {
                    if (!StillCurrent(conversation)) return;

                    _flow.TurnEnded();

                    string heard = (response.UserTranscription ?? "").Trim();
                    ProgressRules.RetryJudgement judgement = ProgressRules.JudgeRetry(target, heard, keyWords);

                    if (judgement.HeardNothing)
                    {
                        OnError?.Invoke("I couldn't make out any speech there - try again a bit closer to the mic.");
                        return;
                    }

                    MistakeEntry entry = LearningProgressService.Instance?.RecordReview(target, judgement);
                    OnRetryJudged?.Invoke(judgement, heard, target, entry);

                    // Back to the conversation, with its question asked again: practice ends here
                    // so that the answer is heard as a turn, not judged as another try. Costs
                    // nothing - the line is built from the last reply and the local voice says it.
                    if (judgement.Passed && _resumeConversationOnPass)
                    {
                        CancelRetry();
                        Speak(RetryPraise.ResumingFrom(_lastReplyText));
                        return;
                    }

                    // Praise costs nothing: the local voice says it, through the same path as
                    // a reply so the mouth moves and it waits for the voice to be free. Without a
                    // voice the card already says it, and the "voice unavailable" warning a
                    // reply would raise is not worth a second time.
                    if (judgement.Passed && _kokoroService != null && _kokoroService.IsInitialized)
                        Speak(RetryPraise.Praise);
                },
                onError: error => { if (StillCurrent(conversation)) OnChatError(error); });
        }

        private IEnumerator MonitorAudioPlayback()
        {
            while (true)
            {
                while (_audioSource != null && _audioSource.isPlaying)
                    yield return null;

                // A piece still being made when the last one ended: a pause, not the end.
                while (_readyChunks.Count == 0 && _chunksToCome > 0)
                    yield return null;

                if (_readyChunks.Count == 0 || _audioSource == null) break;

                // Not CleanupAudio: that stops this coroutine.
                if (_currentClip != null) Destroy(_currentClip);
                _currentClip = _readyChunks.Dequeue();
                _audioSource.clip = _currentClip;
                _audioSource.Play();
            }

            // Audio finished
            InjectAudioPlaybackStopped();
            InjectSpeechStopped();
            InjectTurnCompleted(wasInterrupted: false);

            _audioMonitorCoroutine = null;
            CelebrateWhenDue(_flow.SpeechFinished());
        }

        /// <summary>Celebrates <paramref name="victory"/>, when the flow says one is due.</summary>
        private void CelebrateWhenDue(Victory? victory)
        {
            if (victory.HasValue)
                StartCoroutine(Celebrate(victory.Value));
        }

        /// <summary>
        ///     The victory, a moment after Alice's final line. Dropped if the learner has moved to
        ///     another scenario in that moment.
        /// </summary>
        private IEnumerator Celebrate(Victory victory)
        {
            yield return new WaitForSeconds(0.6f);
            if (_chatService != null && !ConversationFlow.StillValid(victory, _chatService.CurrentScenario))
                yield break;

            var audioMgr = GetComponentInChildren<GeminiAudioManager>();
            audioMgr?.PlayMissionCompleteSound();
            OnAllQuestsCompleted?.Invoke();
            OnMissionVictorySummary?.Invoke(victory.Scenario, victory.AverageScore, victory.Turns, victory.Objectives);
        }

        // ──────────────────── CONVAI EVENT INJECTION ────────────────────

        private void InjectSpeechStarted()
        {
            if (_eventHub == null || string.IsNullOrEmpty(_characterId)) return;

            try
            {
                _eventHub.Publish(CharacterSpeechStateChanged.StartedSpeaking(
                    _characterId, _currentResponseId));
                Debug.Log("[GeminiAvatar] → Injected: CharacterSpeechStateChanged(speaking=true)");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GeminiAvatar] Failed to inject speech started: {ex.Message}");
            }
        }

        private void InjectSpeechStopped()
        {
            if (_eventHub == null || string.IsNullOrEmpty(_characterId)) return;

            try
            {
                _eventHub.Publish(CharacterSpeechStateChanged.StoppedSpeaking(
                    _characterId, _currentResponseId));
                Debug.Log("[GeminiAvatar] → Injected: CharacterSpeechStateChanged(speaking=false)");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GeminiAvatar] Failed to inject speech stopped: {ex.Message}");
            }
        }

        private void InjectAudioPlaybackStarted()
        {
            if (_eventHub == null || string.IsNullOrEmpty(_characterId)) return;

            try
            {
                _eventHub.Publish(CharacterAudioPlaybackStateChanged.Started(_characterId));
                Debug.Log("[GeminiAvatar] → Injected: CharacterAudioPlaybackStateChanged(playing=true)");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GeminiAvatar] Failed to inject audio start: {ex.Message}");
            }
        }

        private void InjectAudioPlaybackStopped()
        {
            if (_eventHub == null || string.IsNullOrEmpty(_characterId)) return;

            try
            {
                _eventHub.Publish(CharacterAudioPlaybackStateChanged.Stopped(_characterId));
                Debug.Log("[GeminiAvatar] → Injected: CharacterAudioPlaybackStateChanged(playing=false)");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GeminiAvatar] Failed to inject audio stop: {ex.Message}");
            }
        }

        private void InjectEmotion(AvatarEmotion emotion, int intensity)
        {
            if (_eventHub == null || string.IsNullOrEmpty(_characterId)) return;

            string emotionStr = emotion switch
            {
                AvatarEmotion.Happy => "happy",
                AvatarEmotion.Sad => "sad",
                AvatarEmotion.Angry => "angry",
                AvatarEmotion.Surprised => "surprise",
                AvatarEmotion.Fear => "fear",
                AvatarEmotion.Disgust => "disgust",
                AvatarEmotion.Thinking => "neutral",
                _ => "neutral"
            };

            try
            {
                _eventHub.Publish(CharacterEmotionChanged.Create(_characterId, emotionStr, intensity));
                Debug.Log($"[GeminiAvatar] → Injected: CharacterEmotionChanged({emotionStr}, {intensity})");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GeminiAvatar] Failed to inject emotion: {ex.Message}");
            }
        }

        private void InjectTtsText(string text)
        {
            if (_eventHub == null || string.IsNullOrEmpty(_characterId)) return;

            try
            {
                _eventHub.Publish(CharacterTtsTextChunk.Create(
                    _characterId, text, chunkIndex: 0, isFinal: true));
                Debug.Log("[GeminiAvatar] → Injected: CharacterTtsTextChunk");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GeminiAvatar] Failed to inject TTS text: {ex.Message}");
            }
        }

        private void InjectTurnCompleted(bool wasInterrupted)
        {
            if (_eventHub == null || string.IsNullOrEmpty(_characterId)) return;

            try
            {
                _eventHub.Publish(CharacterTurnCompleted.Create(
                    _characterId, _characterId, wasInterrupted));
                Debug.Log("[GeminiAvatar] → Injected: CharacterTurnCompleted");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[GeminiAvatar] Failed to inject turn completed: {ex.Message}");
            }
        }

        // ──────────────────── HELPERS ────────────────────

        private void HandleFlowStateChanged(FlowState state)
        {
            AvatarState newState = ToAvatarState(state);
            OnStateChanged?.Invoke(newState);
            Debug.Log($"[GeminiAvatar] State → {newState}");
        }

        /// <summary>
        ///     The flow's state under the name the UI knows. Spelt out rather than cast, so the two
        ///     enums can never drift apart silently.
        /// </summary>
        private static AvatarState ToAvatarState(FlowState state)
        {
            switch (state)
            {
                case FlowState.Recording: return AvatarState.Recording;
                case FlowState.Thinking: return AvatarState.Thinking;
                case FlowState.Speaking: return AvatarState.Speaking;
                default: return AvatarState.Idle;
            }
        }

        private void StopAudioPlayback()
        {
            if (_audioMonitorCoroutine != null)
            {
                StopCoroutine(_audioMonitorCoroutine);
                _audioMonitorCoroutine = null;
            }

            if (_audioSource != null && _audioSource.isPlaying)
                _audioSource.Stop();

            // Whatever was queued belongs to the reply just stopped.
            while (_readyChunks.Count > 0)
                Destroy(_readyChunks.Dequeue());
            _chunksToCome = 0;
        }

        private void CleanupAudio()
        {
            StopAudioPlayback();

            if (_currentClip != null)
            {
                Destroy(_currentClip);
                _currentClip = null;
            }
        }
    }
}
