using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace GeminiAvatar
{
    /// <summary>
    ///     Records audio from the microphone, provides real-time VU meter audio level monitoring,
    ///     allows runtime microphone device selection, and encodes audio as WAV for Gemini multimodal input.
    /// </summary>
    public class SimpleSTTRecorder : MonoBehaviour
    {
        [Header("Recording Settings")]
        [SerializeField] private int _maxRecordingSeconds = 15;
        [SerializeField] private int _sampleRate = 16000;
        [SerializeField] private string _preferredDevice = "";

        [Header("Silence Detection")]
        [Tooltip("Peak amplitude a take must reach to count as speech. Below this the recording is discarded instead of being sent. Raise it in a noisy room, lower it for a quiet microphone.")]
        [Range(0.001f, 0.2f)]
        [SerializeField] private float _silencePeakThreshold = 0.02f;

        private AudioClip _recordingClip;
        private AudioClip _monitorClip;
        private AudioClip _lastDeliveredClip;
        private string _activeDevice;
        private bool _isRecording;
        private float _recordingStartTime;

        /// <summary>
        ///     Shortest take worth sending. A quick tap of the push-to-talk key captures little
        ///     more than the microphone spinning up, which reads as silence downstream.
        /// </summary>
        private const float MinRecordingSeconds = 0.4f;

        /// <summary>Why the last take was thrown away, phrased for the learner. Empty when it was kept.</summary>
        public string LastDiscardReason { get; private set; } = "";

        /// <summary>How far before the buffer end to stop, so GetPosition still reports a real offset.</summary>
        private const float AutoStopMarginSeconds = 0.35f;

        /// <summary>Guards against raising the limit event more than once per take.</summary>
        private bool _maxDurationSignalled;

        private const int SAMPLE_WINDOW = 256;
        private readonly float[] _sampleBuffer = new float[SAMPLE_WINDOW];
        private float _currentAudioLevel;
        private float _levelVelocity;

        /// <summary>Whether a recording is in progress.</summary>
        public bool IsRecording => _isRecording;

        /// <summary>
        ///     The longest take, in seconds. Set before a take starts; the one in progress
        ///     keeps the limit it started with, since the buffer is sized to it.
        /// </summary>
        public int MaxRecordingSeconds
        {
            get => _maxRecordingSeconds;
            set => _maxRecordingSeconds = Mathf.Clamp(value, 5, 150);
        }

        /// <summary>
        ///     Where the speech was in the last take delivered and how much of it was silence,
        ///     for the fluency measure. Empty for a take that was thrown away.
        /// </summary>
        public SpeechTiming LastTakeTiming { get; private set; }

        /// <summary>Seconds into the take in progress; zero when not recording.</summary>
        public float RecordingElapsedSeconds =>
            _isRecording ? Time.realtimeSinceStartup - _recordingStartTime : 0f;

        /// <summary>The currently active microphone device name.</summary>
        public string ActiveDevice => _activeDevice;

        /// <summary>Real-time normalized audio volume level (0.0 to 1.0) for VU meter UI.</summary>
        public float CurrentAudioLevel => _currentAudioLevel;

        /// <summary>Raised when recording state changes.</summary>
        public event Action<bool> OnRecordingStateChanged;

        /// <summary>Raised when the active microphone device changes.</summary>
        public event Action<string> OnDeviceChanged;

        /// <summary>
        ///     Raised when a take reaches the recording limit. The owner should stop the take;
        ///     the audio captured so far is intact and will be delivered normally.
        /// </summary>
        public event Action OnMaxDurationReached;

        private void Start()
        {
            InitializeDevice();
        }

        private void InitializeDevice()
        {
            if (Microphone.devices.Length == 0)
            {
                Debug.LogWarning("[STT] No microphone detected!");
                return;
            }

            // Check saved preference first
            string savedDevice = PlayerPrefs.GetString("GeminiAvatar_PreferredMic", "");
            string targetDevice = !string.IsNullOrEmpty(savedDevice) ? savedDevice : _preferredDevice;

            _activeDevice = null;
            if (!string.IsNullOrEmpty(targetDevice))
            {
                foreach (string dev in Microphone.devices)
                {
                    if (dev.Equals(targetDevice, StringComparison.OrdinalIgnoreCase) ||
                        dev.Contains(targetDevice, StringComparison.OrdinalIgnoreCase))
                    {
                        _activeDevice = dev;
                        break;
                    }
                }
            }

            if (_activeDevice == null)
                _activeDevice = Microphone.devices[0];

            Debug.Log($"[STT] Initialized microphone: {_activeDevice}");
            StartMonitoring();
        }

        private void OnEnable()
        {
            if (!string.IsNullOrEmpty(_activeDevice) && !_isRecording && _monitorClip == null)
            {
                StartMonitoring();
            }
        }

        private void OnDisable()
        {
            // Cancel first: StopMonitoring is a no-op while recording, so ending the take has to
            // happen before the microphone can actually be released.
            CancelRecording(false);
            StopMonitoring();

            if (!string.IsNullOrEmpty(_activeDevice) && Microphone.IsRecording(_activeDevice))
                Microphone.End(_activeDevice);
        }

        private void OnDestroy()
        {
            StopMonitoring();
            if (_isRecording && !string.IsNullOrEmpty(_activeDevice))
            {
                Microphone.End(_activeDevice);
            }
        }

        private void Update()
        {
            UpdateAudioLevel();
            AutoStopBeforeBufferEnds();
        }

        /// <summary>
        ///     Ends the take just before the microphone buffer fills.
        ///
        ///     <para>
        ///     Microphone.Start is called with loop:false, so once the buffer is full capture
        ///     stops and GetPosition reports 0 — which used to be indistinguishable from
        ///     "nothing was recorded", and the learner's whole sentence was discarded in
        ///     silence. Stopping a moment early keeps the position valid.
        ///     </para>
        /// </summary>
        private void AutoStopBeforeBufferEnds()
        {
            if (!_isRecording || _maxDurationSignalled) return;
            if (Time.realtimeSinceStartup - _recordingStartTime < _maxRecordingSeconds - AutoStopMarginSeconds) return;

            _maxDurationSignalled = true;
            Debug.Log($"[STT] Reached the {_maxRecordingSeconds}s limit; closing the take.");
            OnMaxDurationReached?.Invoke();
        }

        private void UpdateAudioLevel()
        {
            if (string.IsNullOrEmpty(_activeDevice))
            {
                _currentAudioLevel = 0f;
                return;
            }

            AudioClip activeClip = _isRecording ? _recordingClip : _monitorClip;
            if (activeClip == null)
            {
                _currentAudioLevel = Mathf.MoveTowards(_currentAudioLevel, 0f, Time.deltaTime * 3f);
                return;
            }

            int micPosition = Microphone.GetPosition(_activeDevice);
            if (micPosition < SAMPLE_WINDOW)
            {
                _currentAudioLevel = Mathf.MoveTowards(_currentAudioLevel, 0f, Time.deltaTime * 3f);
                return;
            }

            int startSample = micPosition - SAMPLE_WINDOW;
            if (activeClip.GetData(_sampleBuffer, startSample))
            {
                float sum = 0f;
                float peak = 0f;
                for (int i = 0; i < SAMPLE_WINDOW; i++)
                {
                    float val = Mathf.Abs(_sampleBuffer[i]);
                    sum += val * val;
                    if (val > peak) peak = val;
                }

                float rms = Mathf.Sqrt(sum / SAMPLE_WINDOW);
                // Amplified for natural responsive UI level meter display
                float rawLevel = Mathf.Clamp01((rms * 6.5f + peak * 1.5f) * 0.5f);

                // Smooth damp for realistic bouncing VU meter needles/bars
                _currentAudioLevel = Mathf.SmoothDamp(_currentAudioLevel, rawLevel, ref _levelVelocity, 0.04f);
            }
            else
            {
                _currentAudioLevel = Mathf.MoveTowards(_currentAudioLevel, 0f, Time.deltaTime * 3f);
            }
        }

        // ──────────────────── DEVICE MANAGEMENT ────────────────────

        /// <summary>Returns list of all connected microphone devices.</summary>
        public string[] GetAvailableDevices()
        {
            return Microphone.devices;
        }

        /// <summary>Switches the active microphone device.</summary>
        public void SelectDevice(string deviceName)
        {
            if (string.IsNullOrEmpty(deviceName) || deviceName == _activeDevice) return;

            bool wasRecording = _isRecording;
            if (_isRecording)
            {
                CancelRecording();
            }

            StopMonitoring();

            _activeDevice = deviceName;
            _preferredDevice = deviceName;
            PlayerPrefs.SetString("GeminiAvatar_PreferredMic", deviceName);
            PlayerPrefs.Save();

            StartMonitoring();
            OnDeviceChanged?.Invoke(_activeDevice);
            Debug.Log($"[STT] Switched active microphone to: {_activeDevice}");
        }

        // ──────────────────── MONITORING LOOP ────────────────────

        private void StartMonitoring()
        {
            // Already running is not an error to start again: a second Microphone.Start over a
            // live monitor restarted the device and orphaned the first clip.
            if (string.IsNullOrEmpty(_activeDevice) || _isRecording || _monitorClip != null) return;

            try
            {
                // Start a lightweight 1-second continuous looping clip for real-time VU monitoring
                _monitorClip = Microphone.Start(_activeDevice, true, 1, _sampleRate);
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[STT] Failed to start microphone monitor: {ex.Message}");
            }
        }

        private void StopMonitoring()
        {
            if (_monitorClip != null && !string.IsNullOrEmpty(_activeDevice))
            {
                if (!_isRecording)
                {
                    Microphone.End(_activeDevice);
                }
                Destroy(_monitorClip);
                _monitorClip = null;
            }
        }

        // ──────────────────── RECORDING PIPELINE ────────────────────

        /// <summary>Starts microphone recording for speech recognition.</summary>
        public void StartRecording()
        {
            if (_isRecording || string.IsNullOrEmpty(_activeDevice)) return;

            // Stop monitor loop before starting recording clip
            StopMonitoring();

            // The clip handed out last turn is no longer referenced by anyone; without this
            // every voice turn leaked its trimmed AudioClip.
            if (_lastDeliveredClip != null)
            {
                Destroy(_lastDeliveredClip);
                _lastDeliveredClip = null;
            }

            LastTakeTiming = default;
            _recordingClip = Microphone.Start(_activeDevice, false, _maxRecordingSeconds, _sampleRate);
            _isRecording = true;
            _recordingStartTime = Time.realtimeSinceStartup;
            _maxDurationSignalled = false;
            OnRecordingStateChanged?.Invoke(true);
            Debug.Log("[STT] Recording started...");
        }

        /// <summary>
        ///     Stops recording and returns the captured audio as a base64 WAV string
        ///     suitable for Gemini multimodal input.
        /// </summary>
        public void StopRecording(Action<string, AudioClip> onComplete)
        {
            if (!_isRecording || string.IsNullOrEmpty(_activeDevice))
            {
                // Said here rather than left over from the last take, whose reason ("too
                // quick") has nothing to do with a microphone that never started.
                LastDiscardReason = "The microphone didn't start - check it under SETTINGS > Mic.";
                onComplete?.Invoke(null, null);
                StartMonitoring();
                return;
            }

            LastDiscardReason = "";

            int lastPosition = Microphone.GetPosition(_activeDevice);
            float heldSeconds = Time.realtimeSinceStartup - _recordingStartTime;
            Microphone.End(_activeDevice);
            _isRecording = false;
            OnRecordingStateChanged?.Invoke(false);

            // A full buffer also reports position 0. Distinguish that from a genuinely empty
            // take by how long the microphone was actually held, and keep the audio.
            if (lastPosition <= 0 && _recordingClip != null && heldSeconds >= _maxRecordingSeconds - AutoStopMarginSeconds * 2f)
            {
                lastPosition = _recordingClip.samples;
                Debug.Log("[STT] Buffer was full; using the complete recording.");
            }

            if (heldSeconds < MinRecordingSeconds)
            {
                Debug.LogWarning($"[STT] Take was only {heldSeconds:F2}s; too short to contain speech.");
                LastDiscardReason = "That was too quick - hold the key while you speak, then release.";
                ReleaseRecordingClip();
                onComplete?.Invoke(null, null);
                StartMonitoring();
                return;
            }

            if (lastPosition <= 0 || _recordingClip == null)
            {
                Debug.LogWarning("[STT] No audio captured.");
                LastDiscardReason = "I didn't get any audio - check the microphone under SETTINGS > Mic.";
                ReleaseRecordingClip();
                onComplete?.Invoke(null, null);
                StartMonitoring();
                return;
            }

            // The take's samples, read once: the silence check, the trimmed clip and the WAV
            // all work from this one copy instead of each reading the clip again.
            int channels = _recordingClip.channels;
            int frequency = _recordingClip.frequency;
            float[] samples = new float[lastPosition * channels];
            _recordingClip.GetData(samples, 0);

            // Silence must never reach the model. Asked to transcribe an empty recording, it
            // does not answer "nothing was said" - it invents a plausible sentence, scores it,
            // and the learner ends up with a turn they never spoke in their progress history.
            float peak = PeakAmplitude(samples);
            if (peak < _silencePeakThreshold)
            {
                Debug.LogWarning($"[STT] Discarding a silent take: peak {peak:F4} < {_silencePeakThreshold:F4}, {heldSeconds:F2}s held.");
                LastDiscardReason = $"I couldn't hear anything (level {peak:F3}). Speak a little louder, or pick a different microphone under SETTINGS > Mic.";
                ReleaseRecordingClip();
                onComplete?.Invoke(null, null);
                StartMonitoring();
                return;
            }

            LastTakeTiming = TurnSkills.MeasureTiming(samples, channels, frequency);

            AudioClip trimmedClip = AudioClip.Create("STT_Recording", lastPosition, channels, frequency, false);
            trimmedClip.SetData(samples, 0);

            byte[] wavBytes = EncodeWav(samples, channels, frequency);
            string base64Wav = Convert.ToBase64String(wavBytes);

            Debug.Log($"[STT] Recording stopped. Duration: {(float)lastPosition / _sampleRate:F2}s, " +
                      $"WAV size: {wavBytes.Length} bytes");

            // The raw buffer is always _maxRecordingSeconds long regardless of what was said,
            // so it is the bigger of the two clips and there is no reason to keep it.
            ReleaseRecordingClip();
            _lastDeliveredClip = trimmedClip;

            // Resume monitor loop
            StartMonitoring();

            onComplete?.Invoke(base64Wav, trimmedClip);
        }

        private void ReleaseRecordingClip()
        {
            if (_recordingClip == null) return;
            Destroy(_recordingClip);
            _recordingClip = null;
        }

        /// <summary>Cancels an in-progress recording without returning data.</summary>
        public void CancelRecording()
        {
            CancelRecording(true);
        }

        /// <param name="resumeMonitoring">
        ///     False when shutting down. OnDisable used to call the resuming version, which ended
        ///     the capture and then immediately started the monitor loop again - leaving the
        ///     microphone live, and the OS recording indicator lit, on a disabled component.
        /// </param>
        public void CancelRecording(bool resumeMonitoring)
        {
            if (!_isRecording) return;

            Microphone.End(_activeDevice);
            _isRecording = false;
            ReleaseRecordingClip();
            OnRecordingStateChanged?.Invoke(false);
            Debug.Log("[STT] Recording cancelled.");

            if (resumeMonitoring) StartMonitoring();
        }

        /// <summary>Loudest absolute sample, 0 for pure silence.</summary>
        private static float PeakAmplitude(float[] data)
        {
            if (data == null) return 0f;

            float peak = 0f;
            for (int i = 0; i < data.Length; i++)
            {
                float magnitude = data[i] < 0f ? -data[i] : data[i];
                if (magnitude > peak) peak = magnitude;
            }
            return peak;
        }

        /// <summary>Encodes an AudioClip as standard 16-bit PCM WAV.</summary>
        public static byte[] EncodeWav(AudioClip clip)
        {
            float[] samples = new float[clip.samples * clip.channels];
            clip.GetData(samples, 0);
            return EncodeWav(samples, clip.channels, clip.frequency);
        }

        /// <summary>Encodes interleaved samples as standard 16-bit PCM WAV.</summary>
        public static byte[] EncodeWav(float[] samples, int channelCount, int frequency)
        {
            using (MemoryStream stream = new MemoryStream(44 + samples.Length * 2))
            using (BinaryWriter writer = new BinaryWriter(stream))
            {
                int sampleCount = samples.Length;
                short channels = (short)channelCount;
                short bitsPerSample = 16;
                int byteRate = frequency * channels * (bitsPerSample / 8);
                short blockAlign = (short)(channels * (bitsPerSample / 8));
                int dataSize = sampleCount * (bitsPerSample / 8);

                // RIFF header
                writer.Write(Encoding.UTF8.GetBytes("RIFF"));
                writer.Write(36 + dataSize);
                writer.Write(Encoding.UTF8.GetBytes("WAVE"));

                // fmt chunk
                writer.Write(Encoding.UTF8.GetBytes("fmt "));
                writer.Write(16);              // chunk size
                writer.Write((short)1);        // PCM format
                writer.Write(channels);
                writer.Write(frequency);
                writer.Write(byteRate);
                writer.Write(blockAlign);
                writer.Write(bitsPerSample);

                // data chunk
                writer.Write(Encoding.UTF8.GetBytes("data"));
                writer.Write(dataSize);

                for (int i = 0; i < sampleCount; i++)
                {
                    short pcm = (short)Mathf.Clamp(samples[i] * 32767f, -32768f, 32767f);
                    writer.Write(pcm);
                }

                return stream.ToArray();
            }
        }
    }
}
