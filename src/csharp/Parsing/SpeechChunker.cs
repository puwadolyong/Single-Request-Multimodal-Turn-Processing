using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace GeminiAvatar
{
    /// <summary>
    ///     Cuts a reply into the pieces the local voice synthesises one after another, so the
    ///     avatar can start speaking the first while the rest are still being made.
    ///
    ///     <para>
    ///     Kokoro was given the whole reply and the avatar stayed silent until all of it was
    ///     synthesised: 2.3 s for a typical 115-character reply on a Ryzen 5 5600, 5.7 s for
    ///     the longest seen - longer than the model had taken to write it. Cut by these rules,
    ///     six of the seven measured replies start speaking 0.9-2.4 s sooner - the seventh opens
    ///     with "Perfect!" and stays whole - with no pause between pieces in any of them.
    ///     </para>
    ///
    ///     <para>
    ///     Cutting at every sentence is not enough, because a piece must still be playing when
    ///     the next one is ready or the avatar stops mid-reply. "Perfect!" lasts under a second
    ///     and the sentence after it took 1.8 s to make, which would have left a pause no
    ///     speaker makes. So a sentence only ends a piece when it lasts long enough to cover the
    ///     next one's synthesis; otherwise it is joined to the next.
    ///     </para>
    ///
    ///     <para>
    ///     Here rather than in the controller because it is a string rule with edge cases and no
    ///     engine in it, so it can be tested without opening Unity.
    ///     </para>
    /// </summary>
    public static class SpeechChunker
    {
        /// <summary>
        ///     Milliseconds of speech per character of text. Measured across seven real replies
        ///     at speed 1.05: 52-55 ms, so the lower figure, which is the cautious one here.
        /// </summary>
        internal const double SpokenMsPerChar = 52;

        /// <summary>
        ///     Synthesis cost, fixed and per character, measured on 4 threads. Multiplied by
        ///     <see cref="SafetyFactor"/> before use, for machines slower than the one measured.
        /// </summary>
        internal const double SynthesisFixedMs = 220;
        internal const double SynthesisMsPerChar = 18.8;
        internal const double SafetyFactor = 1.3;

        /// <summary>
        ///     A full stop, question or exclamation mark followed by space: the only places a
        ///     reply is cut. Not commas - a clause spoken on its own comes out with the falling
        ///     tone of a finished sentence, which is wrong in a voice meant as a model to copy.
        /// </summary>
        private static readonly Regex SentenceEnd = new Regex(@"(?<=[.!?])\s+", RegexOptions.Compiled);

        /// <summary>
        ///     Words whose full stop does not end a sentence. Cutting after "Dr." would make the
        ///     voice say the title with a sentence's falling tone and pause before the name.
        /// </summary>
        private static readonly HashSet<string> Abbreviations = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase)
        {
            "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "mt.", "vs.", "etc.", "e.g.", "i.e.", "a.m.", "p.m.", "no."
        };

        /// <summary>
        ///     The pieces to synthesise, in order. Never empty for non-empty text, and joining
        ///     them with single spaces gives back the reply.
        /// </summary>
        public static List<string> Split(string text)
        {
            var chunks = new List<string>();
            string trimmed = (text ?? "").Trim();
            if (trimmed.Length == 0) return chunks;

            List<string> sentences = Sentences(trimmed);

            string current = sentences[0];
            for (int i = 1; i < sentences.Count; i++)
            {
                string next = sentences[i];
                if (CoversSynthesisOf(current, next))
                {
                    chunks.Add(current);
                    current = next;
                }
                else
                {
                    current = current + " " + next;
                }
            }

            chunks.Add(current);
            return chunks;
        }

        /// <summary>Whether <paramref name="playing"/> lasts longer than making <paramref name="next"/> takes.</summary>
        internal static bool CoversSynthesisOf(string playing, string next)
        {
            double playingMs = playing.Length * SpokenMsPerChar;
            double synthesisMs = (SynthesisFixedMs + next.Length * SynthesisMsPerChar) * SafetyFactor;
            return playingMs >= synthesisMs;
        }

        /// <summary>The reply's sentences, with abbreviations kept attached to what follows.</summary>
        internal static List<string> Sentences(string text)
        {
            var sentences = new List<string>();
            string pending = "";

            foreach (string piece in SentenceEnd.Split(text))
            {
                if (piece.Length == 0) continue;

                pending = pending.Length == 0 ? piece : pending + " " + piece;
                if (!EndsWithAbbreviation(pending))
                {
                    sentences.Add(pending);
                    pending = "";
                }
            }

            if (pending.Length > 0) sentences.Add(pending);
            return sentences;
        }

        private static bool EndsWithAbbreviation(string s)
        {
            int lastSpace = s.LastIndexOf(' ');
            string lastWord = lastSpace < 0 ? s : s.Substring(lastSpace + 1);
            return Abbreviations.Contains(lastWord);
        }
    }
}
