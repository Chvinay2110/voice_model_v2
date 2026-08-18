"""
audio_utils.py

Audio decoding and gentle baseline noise suppression for voiceprint speaker ID.
Optimized for office environments: preserves distant/quiet speakers while stripping
DC drift, low rumble, and ambient background hiss/dead silence.
"""

import base64
import numpy as np
from scipy import signal

SAMPLE_RATE = 16000
MIN_CENTROID_DURATION_S = 1.0  # 1.0 second minimum required for centroid inclusion


def decode_pcm_base64(b64_str: str) -> np.ndarray:
    """Decode base64 encoded 16-bit mono 16kHz PCM into float32 array [-1.0, 1.0]."""
    if not b64_str:
        return np.array([], dtype=np.float32)
    raw_bytes = base64.b64decode(b64_str)
    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0


def clean_audio_for_speaker_id(pcm_float32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Apply gentle baseline noise suppression, DC rumble removal, and level harmonization.

    Tuned specifically for office environments:
    - High-pass filter (> 70 Hz) to eliminate table thumps and electrical hum without clipping voice fundamentals.
    - Soft dynamic noise-floor gating with hangover padding so quiet / distant voices are NOT cut off.
    - RMS level harmonization to bring distant speakers to a clean signal level for ECAPA-TDNN.
    """
    if pcm_float32 is None or len(pcm_float32) == 0:
        return np.array([], dtype=np.float32)

    audio = np.asarray(pcm_float32, dtype=np.float32)

    # 1. Remove DC offset
    audio = audio - np.mean(audio)

    # 2. Gentle 2nd order Butterworth high-pass filter at 70 Hz
    try:
        sos = signal.butter(2, 70.0, btype="highpass", fs=sample_rate, output="sos")
        audio = signal.sosfilt(sos, audio).astype(np.float32)
    except Exception:
        pass

    # 3. Soft Dynamic VAD / Silence Trimming
    frame_len = int(0.025 * sample_rate)  # 25ms = 400 samples
    hop_len = int(0.010 * sample_rate)  # 10ms = 160 samples

    if len(audio) > frame_len:
        # Frame energy (RMS)
        num_frames = 1 + (len(audio) - frame_len) // hop_len
        frames = np.lib.stride_tricks.sliding_window_view(audio[: (num_frames - 1) * hop_len + frame_len], frame_len)[::hop_len]
        rms = np.sqrt(np.mean(frames**2, axis=-1) + 1e-9)

        # Baseline noise floor estimate (15th percentile)
        noise_floor = np.percentile(rms, 15)
        # Non-aggressive threshold: slightly above floor, floor clamp at 0.003 (~ -50 dBFS)
        threshold = max(float(noise_floor) * 1.4, 0.003)

        is_speech_frame = rms > threshold

        # Hangover expansion (keep 150ms = 15 frames before and after speech)
        hangover = 15
        speech_mask = np.copy(is_speech_frame)
        speech_indices = np.where(is_speech_frame)[0]
        for idx in speech_indices:
            start = max(0, idx - hangover)
            end = min(len(speech_mask), idx + hangover + 1)
            speech_mask[start:end] = True

        if np.any(speech_mask):
            # Trim to active speech span (start of first speech frame to end of last speech frame)
            active_frame_indices = np.where(speech_mask)[0]
            start_sample = active_frame_indices[0] * hop_len
            end_sample = min(len(audio), active_frame_indices[-1] * hop_len + frame_len)
            trimmed = audio[start_sample:end_sample]
            if len(trimmed) >= int(0.3 * sample_rate):
                audio = trimmed

    # 4. Gentle volume normalization (bring peak to ~0.75 without clipping)
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        scale = min(0.75 / peak, 4.0)  # Max gain boost of 4x to prevent over-amplifying pure noise
        audio = audio * scale

    return np.clip(audio, -1.0, 1.0).astype(np.float32)