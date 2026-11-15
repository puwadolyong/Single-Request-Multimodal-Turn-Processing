using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using Eitan.SherpaONNXUnity.Runtime.Native;

namespace GeminiAvatar
{
    /// <summary>
    /// High-performance, local offline neural Text-To-Speech service using Kokoro-82M ONNX.
    /// Runs speech synthesis on background threads with zero network delay and generates Unity AudioClips in milliseconds.
    /// </summary>
    public class KokoroTTSService : MonoBehaviour
    {
        /// <summary>Which model speaks. Kokoro is English only; Supertonic 3 reads Japanese text itself.</summary>
        public enum Engine { Kokoro, Supertonic }

        [Header("Engine")]
        // English by default. The controller adds this component at run time, so a scene cannot
        // set it: the Japanese edition is chosen once, on the chat service, and Awake follows it.
        [Tooltip("Kokoro-82M (English) or Supertonic 3 (31 languages, used here for Japanese).")]
        [SerializeField] private Engine _engine = Engine.Kokoro;

        [Tooltip("Relative path inside StreamingAssets to the Supertonic 3 model directory.")]
        [SerializeField] private string _supertonicSubPath = "Supertonic-3";

        [Tooltip("Language Supertonic speaks, as its code: ja, en, ko ...")]
        [SerializeField] private string _language = "en";

        [Header("Kokoro Model Settings")]
        [Tooltip("Relative path inside StreamingAssets to the Kokoro model directory.")]
        [SerializeField] private string _modelSubPath = "Kokoro-82M/kokoro-en-v0_19";

        [Header("Voice & Playback Settings")]
        [Range(0.5f, 2.0f)]
        [Tooltip("Speaking speed multiplier (1.0 = normal, 1.05 = crisp/natural, 1.25 = fast).")]
        [SerializeField] private float _speechSpeed = 1.05f;

        [Tooltip("Speaker ID (0: af_bella, 1: af_sarah, 2: af_nicole, 3: af_sky, 4: am_adam, 5: am_michael, 6: bf_emma, 7: bf_isabella, 8: bm_george, 9: bm_lewis).")]
        [SerializeField] private int _speakerId = 0;

        [Tooltip("Number of CPU threads to use for ONNX inference.")]
        [SerializeField] private int _numThreads = 4;

        private OfflineTts _tts;

        /// <summary>
        ///     Serialises native calls against <see cref="_tts"/>.
        ///
        ///     <para>
        ///     Synthesis runs on a worker thread while Dispose runs on the main thread at
        ///     OnDestroy, so quitting play mode mid-sentence could free the native handle while
        ///     the worker was still inside Generate — a use-after-free that takes the Editor
        ///     down rather than raising a managed exception. The same gate stops two overlapping
        ///     requests from entering Generate on one handle.
        ///     </para>
        /// </summary>
        private readonly object _ttsGate = new object();
        private bool _isInitialized;
        private bool _isProcessing;
        private string _resolvedModelDir;

        public Engine ActiveEngine => _engine;
        public string Language { get => _language; set => _language = value ?? ""; }
        public bool IsInitialized => _isInitialized;
        public bool IsProcessing => _isProcessing;
        public float SpeechSpeed { get => _speechSpeed; set => _speechSpeed = Mathf.Clamp(value, 0.5f, 2.0f); }
        public int SpeakerId { get => _speakerId; set => _speakerId = Mathf.Clamp(value, 0, SpeakerNames.Length - 1); }
        public int NumThreads { get => _numThreads; set => _numThreads = Mathf.Clamp(value, 1, 16); }

        /// <summary>
        ///     The bundled model's voices, indexed by speaker id. Eleven, in the model's order;
        ///     see <see cref="GeminiAvatarController.KokoroVoiceOption"/> for how a ten-name list
        ///     here once put the wrong voice behind every id but 0.
        /// </summary>
        public static readonly string[] SpeakerNames = new string[]
        {
            "af (US Female - default blend)",
            "af_bella (US Female - Warm)",
            "af_nicole (US Female - Soft)",
            "af_sarah (US Female - Clear)",
            "af_sky (US Female - Bright)",
            "am_adam (US Male - Deep)",
            "am_michael (US Male - Clear)",
            "bf_emma (UK Female)",
            "bf_isabella (UK Female)",
            "bm_george (UK Male)",
            "bm_lewis (UK Male)"
        };

        /// <summary>
        ///     Whether the model is still being read. False both before loading starts and once
        ///     it has finished, so callers check this together with <see cref="IsInitialized"/>.
        /// </summary>
        public bool IsLoading { get; private set; }

        /// <summary>
        ///     Synthesising a sentence right now. One at a time: a request made meanwhile is
        ///     refused, so a caller that has to be heard waits for this to clear first.
        /// </summary>
        public bool IsBusy => _isProcessing;

        private void Awake()
        {
            // The Japanese edition: Supertonic, speaking Japanese. ScenarioLanguage is set by the
            // chat service, which wakes before everything else.
            if (ScenarioLanguage.IsJapanese)
            {
                _engine = Engine.Supertonic;
                _language = "ja";
            }

            BeginInitialise();
        }

        /// <summary>
        ///     Reads the model on a worker thread.
        ///
        ///     <para>
        ///     It used to be read in Awake, on the main thread. The file is 330 MB, so the
        ///     application froze on a blank window for several seconds at every launch with
        ///     nothing on screen to say why - Unity cannot draw a frame while the main thread is
        ///     inside a native call, which is also why a loading screen alone would not have
        ///     helped. Nothing can be drawn until this moves off that thread.
        ///     </para>
        ///
        ///     <para>
        ///     The Unity APIs it needs are read here first. streamingAssetsPath and
        ///     processorCount throw when touched from a worker, and finding that out at run time
        ///     on somebody else's machine is not the way to learn it.
        ///     </para>
        /// </summary>
        private void BeginInitialise()
        {
            if (_isInitialized || IsLoading) return;

            string baseDir = Application.streamingAssetsPath;
            int threads = _numThreads > 0 ? _numThreads : Math.Max(2, SystemInfo.processorCount / 2);

            IsLoading = true;
            Task.Run(() =>
            {
                try
                {
                    InitializeTts(baseDir, threads);
                }
                finally
                {
                    IsLoading = false;
                }
            });
        }

        private void OnDestroy()
        {
            DisposeTts();
        }

        /// <summary>
        /// Initializes the Kokoro-82M ONNX model from StreamingAssets.
        /// </summary>
        public bool InitializeTts() =>
            InitializeTts(Application.streamingAssetsPath,
                          _numThreads > 0 ? _numThreads : Math.Max(2, SystemInfo.processorCount / 2));

        /// <summary>
        ///     The real loader. Takes what it needs from Unity as arguments so it can run on a
        ///     worker thread, where touching the engine would throw.
        /// </summary>
        private bool InitializeTts(string baseDir, int threads)
        {
            if (_isInitialized && _tts != null) return true;
            if (_engine == Engine.Supertonic) return InitializeSupertonic(baseDir, threads);

            try
            {
                string candidate1 = Path.Combine(baseDir, _modelSubPath);
                string candidate2 = Path.Combine(baseDir, "Kokoro-82M");
                string candidate3 = Path.Combine(baseDir, "Kokoro-82M", "kokoro-en-v0_19");

                string targetDir = null;
                if (Directory.Exists(candidate1) && File.Exists(Path.Combine(candidate1, "model.onnx")))
                    targetDir = candidate1;
                else if (Directory.Exists(candidate3) && File.Exists(Path.Combine(candidate3, "model.onnx")))
                    targetDir = candidate3;
                else if (Directory.Exists(candidate2) && File.Exists(Path.Combine(candidate2, "model.onnx")))
                    targetDir = candidate2;

                if (string.IsNullOrEmpty(targetDir))
                {
                    Debug.LogWarning($"[KokoroTTS] Model directory not found under '{_modelSubPath}'. Make sure files exist in StreamingAssets.");
                    return false;
                }

                _resolvedModelDir = targetDir;
                string modelPath = Path.Combine(targetDir, "model.onnx");
                string voicesPath = Path.Combine(targetDir, "voices.bin");
                string tokensPath = Path.Combine(targetDir, "tokens.txt");
                string dataDir = Path.Combine(targetDir, "espeak-ng-data");

                if (!File.Exists(modelPath) || !File.Exists(tokensPath))
                {
                    Debug.LogWarning($"[KokoroTTS] Missing required Kokoro files in '{targetDir}'.");
                    return false;
                }

                var config = new OfflineTtsConfig(true);
                config.Model.NumThreads = threads;
                config.Model.Provider = "cpu";
                config.Model.Kokoro.Model = modelPath;
                config.Model.Kokoro.Tokens = tokensPath;
                config.Model.Kokoro.DataDir = Directory.Exists(dataDir) ? dataDir : "";
                config.Model.Kokoro.Voices = File.Exists(voicesPath) ? voicesPath : "";
                config.Model.Kokoro.LengthScale = 1.0f;

                _tts = new OfflineTts(config);
                _isInitialized = true;
                Debug.Log($"<color=#4ADE80>[KokoroTTS] Kokoro-82M ONNX initialized successfully!</color> SampleRate: {_tts.SampleRate}Hz, Speakers: {_tts.NumSpeakers}");
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[KokoroTTS] Initialization error: {ex.Message}\n{ex.StackTrace}");
                _isInitialized = false;
                return false;
            }
        }

        /// <summary>
        ///     Supertonic 3 through the same sherpa-onnx library. It needs sherpa-onnx 1.13.2 or
        ///     later; the package's own 1.13.0 binaries are replaced with 1.13.8 in this project,
        ///     whose C API structs are unchanged from 1.13.0, so the package's C# still matches.
        /// </summary>
        private bool InitializeSupertonic(string baseDir, int threads)
        {
            try
            {
                string dir = Path.Combine(baseDir, _supertonicSubPath);
                string[] files =
                {
                    "duration_predictor.int8.onnx", "text_encoder.int8.onnx", "vector_estimator.int8.onnx",
                    "vocoder.int8.onnx", "tts.json", "unicode_indexer.bin", "voice.bin"
                };
                foreach (string f in files)
                {
                    if (!File.Exists(Path.Combine(dir, f)))
                    {
                        Debug.LogWarning($"[SupertonicTTS] Missing '{f}' in '{dir}'.");
                        return false;
                    }
                }

                _resolvedModelDir = dir;
                var config = new OfflineTtsConfig(true);
                config.Model.NumThreads = threads;
                config.Model.Provider = "cpu";
                config.Model.Supertonic.DurationPredictor = Path.Combine(dir, files[0]);
                config.Model.Supertonic.TextEncoder = Path.Combine(dir, files[1]);
                config.Model.Supertonic.VectorEstimator = Path.Combine(dir, files[2]);
                config.Model.Supertonic.Vocoder = Path.Combine(dir, files[3]);
                config.Model.Supertonic.TtsJson = Path.Combine(dir, files[4]);
                config.Model.Supertonic.UnicodeIndexer = Path.Combine(dir, files[5]);
                config.Model.Supertonic.VoiceStyle = Path.Combine(dir, files[6]);

                _tts = new OfflineTts(config);
                _isInitialized = true;
                Debug.Log($"<color=#4ADE80>[SupertonicTTS] Supertonic 3 initialized.</color> SampleRate: {_tts.SampleRate}Hz, Speakers: {_tts.NumSpeakers}, Language: {_language}");
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SupertonicTTS] Initialization error: {ex.Message}\n{ex.StackTrace}");
                _isInitialized = false;
                return false;
            }
        }

        /// <summary>Supertonic is told the language per request; the callback only lets synthesis continue.</summary>
        private OfflineTtsGeneratedAudio GenerateSupertonic(string text, float speed, int sid)
        {
            var gen = new OfflineTtsGenerationConfig { Speed = speed, Sid = sid };
            gen.Extra["lang"] = string.IsNullOrEmpty(_language) ? "en" : _language;
            return _tts.GenerateWithConfig(text, gen, (samples, n, progress, arg) => 1);
        }

        /// <summary>
        /// Synthesizes English text into a Unity AudioClip locally using Kokoro-82M.
        /// </summary>
        public void GenerateSpeech(string text, Action<AudioClip> onSuccess, Action<string> onError)
        {
            if (!_isInitialized || _tts == null)
            {
                if (!InitializeTts())
                {
                    onError?.Invoke("Kokoro TTS model is not initialized.");
                    return;
                }
            }

            if (string.IsNullOrWhiteSpace(text))
            {
                onError?.Invoke("Text is empty.");
                return;
            }

            // _isProcessing was set but never tested, so a second request could start while the
            // first was still running.
            if (_isProcessing)
            {
                onError?.Invoke("Kokoro TTS is already synthesizing speech.");
                return;
            }

            // Clean text tags
            string cleanText = text;
            if (cleanText.Contains("[EMOTION:"))
            {
                int tagEnd = cleanText.IndexOf("]");
                if (tagEnd >= 0)
                    cleanText = cleanText.Substring(tagEnd + 1).Trim();
            }
            cleanText = cleanText.Replace("*", "").Replace("#", "").Trim();
            if (string.IsNullOrWhiteSpace(cleanText)) cleanText = "Hello!";

            float speed = _speechSpeed;
            int sid = _speakerId;

            _isProcessing = true;

            // Run ONNX neural inference asynchronously on background worker thread
            Task.Run(() =>
            {
                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();

                    OfflineTtsGeneratedAudio audio;
                    lock (_ttsGate)
                    {
                        // Disposed while this request was queued: bail out instead of calling
                        // into a freed native handle.
                        if (_tts == null)
                        {
                            UnityMainThreadDispatcher.Enqueue(() =>
                            {
                                _isProcessing = false;
                                onError?.Invoke("Kokoro TTS was shut down before synthesis finished.");
                            });
                            return;
                        }

                        audio = _engine == Engine.Supertonic
                            ? GenerateSupertonic(cleanText, speed, sid)
                            : _tts.Generate(cleanText, speed, sid);
                    }

                    sw.Stop();

                    if (audio == null || audio.NumSamples == 0)
                    {
                        UnityMainThreadDispatcher.Enqueue(() =>
                        {
                            _isProcessing = false;
                            onError?.Invoke("Kokoro TTS produced no audio samples.");
                        });
                        return;
                    }

                    float[] samples = audio.Samples;
                    int sampleRate = audio.SampleRate;
                    int numSamples = audio.NumSamples;
                    audio.Dispose();

                    // Switch back to Unity main thread to create AudioClip
                    UnityMainThreadDispatcher.Enqueue(() =>
                    {
                        _isProcessing = false;
                        try
                        {
                            AudioClip clip = AudioClip.Create($"Kokoro_Audio_{DateTime.Now.Ticks}", numSamples, 1, sampleRate, false);
                            clip.SetData(samples, 0);
                            Debug.Log($"<color=#38BDF8>[KokoroTTS] Generated {numSamples} samples ({clip.length:F2}s) in {sw.ElapsedMilliseconds}ms!</color>");
                            onSuccess?.Invoke(clip);
                        }
                        catch (Exception ex)
                        {
                            onError?.Invoke($"Failed to create AudioClip: {ex.Message}");
                        }
                    });
                }
                catch (Exception ex)
                {
                    UnityMainThreadDispatcher.Enqueue(() =>
                    {
                        _isProcessing = false;
                        onError?.Invoke($"Kokoro TTS generation error: {ex.Message}");
                    });
                }
            });
        }

        private void DisposeTts()
        {
            // Taking the same gate the worker holds: if synthesis is in flight this waits the
            // few hundred milliseconds for it to finish rather than freeing the handle underneath it.
            lock (_ttsGate)
            {
                if (_tts != null)
                {
                    _tts.Dispose();
                    _tts = null;
                }
            }

            _isInitialized = false;
            _isProcessing = false;
        }
    }

    /// <summary>
    /// Helper to safely dispatch callbacks to Unity's main thread.
    /// </summary>
    public class UnityMainThreadDispatcher : MonoBehaviour
    {
        private static readonly System.Collections.Generic.Queue<Action> _executionQueue = new System.Collections.Generic.Queue<Action>();
        private static UnityMainThreadDispatcher _instance;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Initialize()
        {
            if (_instance == null)
            {
                var go = new GameObject("UnityMainThreadDispatcher");
                _instance = go.AddComponent<UnityMainThreadDispatcher>();
                DontDestroyOnLoad(go);
            }
        }

        public static void Enqueue(Action action)
        {
            if (action == null) return;
            lock (_executionQueue)
            {
                _executionQueue.Enqueue(action);
            }
        }

        private void Update()
        {
            lock (_executionQueue)
            {
                while (_executionQueue.Count > 0)
                {
                    _executionQueue.Dequeue()?.Invoke();
                }
            }
        }
    }
}
