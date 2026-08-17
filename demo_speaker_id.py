"""
demo_speaker_id.py

Standalone Interactive Demo for Speaker Identification & Voiceprint Recognition.
Uses SpeechBrain's state-of-the-art ECAPA-TDNN model (trained on VoxCeleb).

How to run:
    python demo_speaker_id.py
"""

import sys
import time
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
ENROLL_DURATION = 4.0  # seconds per speaker enrollment
TEST_DURATION = 4.0    # seconds for each test speech turn

print("=" * 60)
print("🎙️  ECAPA-TDNN SPEAKER IDENTIFICATION DEMO")
print("=" * 60)
print("Loading pre-trained ECAPA-TDNN acoustic model...")

# Load SpeechBrain ECAPA-TDNN encoder (using COPY on Windows to avoid symlink privilege errors)
try:
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    import warnings
    warnings.filterwarnings("ignore")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        local_strategy=LocalStrategy.COPY_SKIP_CACHE
    )
    print("✅ Model loaded successfully (192-dimensional embeddings).")
except Exception as e:
    print("❌ Failed to load SpeechBrain model:", e)
    sys.exit(1)


def record_audio(duration, prompt_msg):
    print(f"\n👉 {prompt_msg}")
    input("   Press [ENTER] when ready to speak...")
    print(f"   🔴 RECORDING ({duration}s)... Speak clearly!")
    
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("   ⏹️ Recording finished.")
    
    # Flatten to 1D
    audio_flat = audio.squeeze()
    
    # Simple RMS volume check
    rms = np.sqrt(np.mean(audio_flat**2))
    if rms < 0.01:
        print("   ⚠️ Warning: Audio level is very low. Please check your mic.")
    
    return audio_flat


def extract_embedding(audio_data):
    """Extracts normalized 192-d embedding vector using ECAPA-TDNN."""
    tensor = torch.tensor(audio_data).unsqueeze(0)
    with torch.no_grad():
        emb = classifier.encode_batch(tensor)
        emb = F.normalize(emb.squeeze(), p=2, dim=-1)
    return emb


# Store enrolled speaker voiceprints: { "Name": embedding_tensor }
voiceprints = {}


def enroll_speaker(name):
    audio = record_audio(ENROLL_DURATION, f"Enroll voice for '{name}' (speak a random 4s sentence)")
    emb = extract_embedding(audio)
    voiceprints[name] = emb
    print(f"   ✅ Voiceprint registered for '{name}'.")


def main():
    print("\n--- STEP 1: VOICEPRINT ENROLLMENT ---")
    speaker1_name = input("Enter name for Speaker 1 (e.g. Vinay): ").strip() or "Speaker 1"
    enroll_speaker(speaker1_name)

    speaker2_name = input("\nEnter name for Speaker 2 (e.g. Sarah / Alex): ").strip() or "Speaker 2"
    enroll_speaker(speaker2_name)

    print("\n" + "=" * 60)
    print("--- STEP 2: LIVE SPEAKER IDENTIFICATION TEST ---")
    print("Speak any sentence, and the AI will predict who spoke with confidence.")
    print("=" * 60)

    while True:
        audio = record_audio(TEST_DURATION, "Test Identification")
        test_emb = extract_embedding(audio)

        scores = {}
        for name, enrolled_emb in voiceprints.items():
            # Cosine similarity between normalized embeddings
            sim = torch.dot(test_emb, enrolled_emb).item()
            scores[name] = max(0.0, sim)

        best_speaker = max(scores, key=scores.get)
        best_score = scores[best_speaker]

        # Convert similarity to intuitive percentage confidence (typically 0.4 to 0.9 range for matches)
        conf_pct = min(99.9, max(1.0, (best_score - 0.2) / 0.65 * 100)) if best_score > 0.3 else 10.0

        print("\n" + "─" * 45)
        print(f"🎯 PREDICTED SPEAKER:  >>> {best_speaker} <<<")
        print(f"📊 Match Confidence :  {conf_pct:.1f}% (Cosine Score: {best_score:.3f})")
        print("🔍 Similarity Breakdown:")
        for name, score in scores.items():
            bar = "█" * int(score * 25)
            print(f"   • {name:15}: {score:.3f} | {bar}")
        print("─" * 45)

        again = input("\nTest another sample? (Press [ENTER] to continue, or 'q' to quit): ").strip().lower()
        if again == 'q':
            break

    print("\nDemo finished. Thank you!")


if __name__ == "__main__":
    main()
