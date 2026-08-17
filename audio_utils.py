"""
audio_utils.py

Shared audio decode + preprocess step used by both speaker_engine.py and
whisper_engine.py, so an uploaded utterance is decoded and cleaned exactly
once per /api/process call instead of twice.
"""

import io

import numpy as np
import soundfile as sf

from resemblyzer import preprocess_wav  # VAD trim / resample to 16kHz / volume-norm

MIN_AUDIO_SECONDS = 0.4


def load_and_preprocess(wav_bytes, min_audio_seconds=MIN_AUDIO_SECONDS):
    """Decode a WAV file's bytes, downmix to mono, and run it through
    Resemblyzer's preprocess_wav (VAD trim + resample to 16kHz + volume
    normalize). Returns a 16kHz mono float32 numpy array, or None if the
    audio is too short/silent for either embedding or transcription to be
    worth attempting.
    """
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr <= 0 or len(data) == 0:
        return None
    duration_s = len(data) / float(sr)
    if duration_s < min_audio_seconds:
        return None

    wav16k = preprocess_wav(data, source_sr=sr)
    if wav16k is None or len(wav16k) < int(0.3 * 16000):
        return None  # VAD decided there was basically no speech here

    return np.asarray(wav16k, dtype=np.float32)