using NUnit.Framework;

namespace GeminiAvatar.Tests
{
    /// <summary>
    ///     Contract tests for <see cref="SpeechChunker.Split"/>.
    ///
    ///     The replies are real ones from the 2026-09 telemetry runs, so the cuts asserted here
    ///     are the cuts learners hear.
    /// </summary>
    [TestFixture]
    public class SpeechChunkerTests
    {
        [Test]
        public void TwoSentencesOfReasonableLengthAreSpokenSeparately()
        {
            var chunks = SpeechChunker.Split(
                "Please use a complete sentence. What is the main purpose of your visit today?");

            Assert.That(chunks, Is.EqualTo(new[]
            {
                "Please use a complete sentence.",
                "What is the main purpose of your visit today?"
            }));
        }

        [Test]
        public void AShortOpenerIsJoinedToTheSentenceItCouldNotCover()
        {
            // "Perfect!" is over before the long sentence after it could be made.
            const string reply = "Perfect! Please tap or insert your card right here, and would you like me to print your receipt or email it to you?";

            Assert.That(SpeechChunker.Split(reply), Is.EqualTo(new[] { reply }));
        }

        [Test]
        public void ALongReplyStartsWithAPieceThatCoversWhatFollows()
        {
            var chunks = SpeechChunker.Split(
                "Of course. For the pain and sore throat, you should take one paracetamol tablet every four to six hours as needed, but no more than four tablets in twenty-four hours. " +
                "For the lozenges, dissolve one slowly in your mouth every three hours. " +
                "Do you have any questions about these instructions?");

            Assert.That(chunks.Count, Is.EqualTo(3));
            Assert.That(chunks[0], Does.StartWith("Of course. For the pain"));
            Assert.That(chunks[2], Is.EqualTo("Do you have any questions about these instructions?"));
        }

        [Test]
        public void JoiningThePiecesGivesBackTheReply()
        {
            const string reply = "I understand you have cash, but I cannot let you enter without the name of your hotel and proof of a return flight. Please show me your booking confirmation or your return ticket.";

            Assert.That(string.Join(" ", SpeechChunker.Split(reply)), Is.EqualTo(reply));
        }

        [Test]
        public void ATitleIsNeverCutFromTheNameAfterIt()
        {
            var chunks = SpeechChunker.Split(
                "Your appointment this afternoon is with Dr. Patel on the second floor. Please take a seat in the waiting area until she calls you in.");

            Assert.That(chunks, Has.None.EndsWith("Dr."));
        }

        [Test]
        public void ASingleSentenceIsOnePiece()
        {
            Assert.That(SpeechChunker.Split("  Sure thing, one iced latte with oat milk!  "),
                        Is.EqualTo(new[] { "Sure thing, one iced latte with oat milk!" }));
        }

        [Test]
        public void DecimalsAreNotSentenceEnds()
        {
            Assert.That(SpeechChunker.Split("That will be 3.50 for the small one, please.").Count, Is.EqualTo(1));
        }

        [TestCase(null)]
        [TestCase("")]
        [TestCase("   ")]
        public void NothingToSayGivesNoPieces(string text)
        {
            Assert.That(SpeechChunker.Split(text), Is.Empty);
        }
    }
}
