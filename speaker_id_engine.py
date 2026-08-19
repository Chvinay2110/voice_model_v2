"""
speaker_id_engine.py

ECAPA-TDNN voiceprint engine with dynamic centroid learning, user management,
and SQLite persistence.
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import db

log = logging.getLogger("speaker_id_engine")
warnings.filterwarnings("ignore")

# ── Lazy-loaded model singleton ─────────────────────────────────────────────

_classifier = None


def _load_model():
    """Load the SpeechBrain ECAPA-TDNN model once, lazily."""
    global _classifier
    if _classifier is not None:
        return _classifier

    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading ECAPA-TDNN model on %s ...", device)

    _classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY_SKIP_CACHE,
    )
    log.info("ECAPA-TDNN model loaded (192-d embeddings).")
    return _classifier


# ── Multi-Sample User Profiles ───────────────────────────────────────────────
_profiles: Dict[str, dict] = {}


def init_profiles_from_db():
    """Load profiles and centroid embeddings from SQLite database."""
    global _profiles
    db.init_db()
    db_profiles = db.get_all_profiles()
    for p in db_profiles:
        name = p["name"]
        centroid_tensor = None
        if p.get("centroid"):
            centroid_tensor = torch.tensor(p["centroid"], dtype=torch.float32)
            centroid_tensor = F.normalize(centroid_tensor, p=2, dim=-1)

        _profiles[name] = {
            "embeddings": [centroid_tensor] if centroid_tensor is not None else [],
            "centroid": centroid_tensor,
            "enrolled_centroid": centroid_tensor,
            "avatar_b64": p.get("avatar_b64"),
            "company_name": p.get("company_name") or "",
        }
    log.info("Loaded %d profiles from SQLite database.", len(_profiles))


from audio_utils import clean_audio_for_speaker_id


def extract_embedding(pcm_float32_16k: np.ndarray) -> torch.Tensor:
    """Clean audio with office noise baseline and extract normalized 192-d embedding."""
    cleaned_pcm = clean_audio_for_speaker_id(pcm_float32_16k)
    if len(cleaned_pcm) < int(0.3 * 16000):
        cleaned_pcm = pcm_float32_16k
    model = _load_model()
    tensor = torch.tensor(cleaned_pcm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(tensor)
        emb = F.normalize(emb.squeeze(), p=2, dim=-1)
    return emb


def create_user(speaker_name: str, avatar_b64: Optional[str] = None, company_name: Optional[str] = None) -> dict:
    """Create or update a user profile with optional avatar image."""
    name = speaker_name.strip()
    if not name:
        raise ValueError("User name cannot be empty")
    if name not in _profiles:
        _profiles[name] = {
            "embeddings": [],
            "centroid": None,
            "enrolled_centroid": None,
            "avatar_b64": avatar_b64,
            "company_name": (company_name or "").strip(),
        }
        log.info("Created user profile: '%s'", name)
    elif avatar_b64:
        _profiles[name]["avatar_b64"] = avatar_b64
    if company_name is not None:
        _profiles[name]["company_name"] = company_name.strip()

    # Save to SQLite
    centroid_list = _profiles[name]["centroid"].tolist() if _profiles[name]["centroid"] is not None else None
    db.upsert_profile(
        name=name,
        avatar_b64=_profiles[name].get("avatar_b64"),
        company_name=_profiles[name].get("company_name"),
        samples_count=len(_profiles[name]["embeddings"]),
        centroid_list=centroid_list,
    )

    return {
        "speaker": name,
        "samples_count": len(_profiles[name]["embeddings"]),
        "has_voiceprint": _profiles[name]["centroid"] is not None,
        "avatar_b64": _profiles[name].get("avatar_b64"),
        "company_name": _profiles[name].get("company_name", ""),
    }


def set_user_avatar(speaker_name: str, avatar_b64: str) -> dict:
    """Set or update a user's profile avatar image (base64 data URL)."""
    lower_map = {k.lower().strip(): k for k in _profiles.keys()}
    canon_name = lower_map.get((speaker_name or "").lower().strip(), speaker_name)
    name = (canon_name or "").strip()
    if not name:
        raise ValueError("Speaker name cannot be empty")
    if name not in _profiles:
        create_user(name, avatar_b64=avatar_b64)
    else:
        _profiles[name]["avatar_b64"] = avatar_b64
        centroid_list = _profiles[name]["centroid"].tolist() if _profiles[name]["centroid"] is not None else None
        db.upsert_profile(
            name=name,
            avatar_b64=avatar_b64,
            company_name=_profiles[name].get("company_name"),
            samples_count=len(_profiles[name]["embeddings"]),
            centroid_list=centroid_list,
        )
    return {"speaker": name, "avatar_b64": avatar_b64}


def delete_user(speaker_name: str) -> dict:
    """Delete a user profile and their voiceprint centroid."""
    lower_map = {k.lower().strip(): k for k in _profiles.keys()}
    canon_name = lower_map.get((speaker_name or "").lower().strip(), speaker_name)
    name = (canon_name or "").strip()
    if name in _profiles:
        del _profiles[name]
        log.info("Deleted user profile: '%s'", name)
    db.delete_profile(name)
    return {"speaker": name, "status": "deleted"}


def sync_all_user_centroids(speaker_to_embeddings: Dict[str, List[torch.Tensor]]):
    """Rebuild all user centroids from the provided turn embedding collections."""
    lower_grouped = {k.lower().strip(): v for k, v in speaker_to_embeddings.items()}
    for name in list(_profiles.keys()):
        enrolled = _profiles[name].get("enrolled_centroid")
        embs = lower_grouped.get(name.lower().strip(), [])
        all_embs = [enrolled] + embs if enrolled is not None else embs

        if all_embs:
            _profiles[name]["embeddings"] = all_embs
            stacked = torch.stack(all_embs, dim=0)
            mean_emb = torch.mean(stacked, dim=0)
            _profiles[name]["centroid"] = F.normalize(mean_emb, p=2, dim=-1)
        else:
            _profiles[name]["embeddings"] = []
            _profiles[name]["centroid"] = None

        centroid_list = _profiles[name]["centroid"].tolist() if _profiles[name]["centroid"] is not None else None
        db.upsert_profile(
            name=name,
            avatar_b64=_profiles[name].get("avatar_b64"),
            company_name=_profiles[name].get("company_name"),
            samples_count=len(_profiles[name]["embeddings"]),
            centroid_list=centroid_list,
        )

    for name, embs in speaker_to_embeddings.items():
        if name and name not in _profiles:
            create_user(name)
            if embs:
                _profiles[name]["embeddings"] = embs
                stacked = torch.stack(embs, dim=0)
                mean_emb = torch.mean(stacked, dim=0)
                _profiles[name]["centroid"] = F.normalize(mean_emb, p=2, dim=-1)
                db.upsert_profile(
                    name=name,
                    avatar_b64=_profiles[name].get("avatar_b64"),
                    company_name=_profiles[name].get("company_name"),
                    samples_count=len(_profiles[name]["embeddings"]),
                    centroid_list=_profiles[name]["centroid"].tolist(),
                )


def update_user_centroid(name: str, embeddings: list):
    """Update a single user's centroid from a list of embeddings."""
    lower_map = {k.lower().strip(): k for k in _profiles.keys()}
    canon_name = lower_map.get((name or "").lower().strip(), name)
    if canon_name not in _profiles:
        return
    name = canon_name

    enrolled = _profiles[name].get("enrolled_centroid")

    if embeddings:
        all_embs = [enrolled] + embeddings if enrolled is not None else embeddings
        _profiles[name]["embeddings"] = all_embs
        stacked = torch.stack(all_embs, dim=0)
        mean_emb = torch.mean(stacked, dim=0)
        _profiles[name]["centroid"] = F.normalize(mean_emb, p=2, dim=-1)
    else:
        if enrolled is not None:
            _profiles[name]["embeddings"] = [enrolled]
            _profiles[name]["centroid"] = enrolled
        else:
            _profiles[name]["embeddings"] = []
            _profiles[name]["centroid"] = None

    centroid_list = _profiles[name]["centroid"].tolist() if _profiles[name]["centroid"] is not None else None
    db.upsert_profile(
        name=name,
        avatar_b64=_profiles[name].get("avatar_b64"),
        company_name=_profiles[name].get("company_name"),
        samples_count=len(_profiles[name]["embeddings"]),
        centroid_list=centroid_list,
    )


def enroll_user_long_audio(speaker_name: str, pcm_float32_16k: np.ndarray) -> dict:
    """Enroll a user by processing the entire continuous recorded audio clip (e.g. 10s–60s reading)
    directly through ECAPA-TDNN's Attentive Statistics Pooling."""
    name = speaker_name.strip()
    if not name:
        raise ValueError("User name required")
    if name not in _profiles:
        create_user(name)

    pcm_len = len(pcm_float32_16k)
    sample_rate = 16000
    duration_s = pcm_len / sample_rate

    # Extract 192-d centroid directly from the entire raw continuous audio clip
    centroid = extract_embedding(pcm_float32_16k)

    _profiles[name]["embeddings"] = [centroid]
    _profiles[name]["centroid"] = centroid

    db.upsert_profile(
        name=name,
        avatar_b64=_profiles[name].get("avatar_b64"),
        company_name=_profiles[name].get("company_name"),
        samples_count=1,
        centroid_list=centroid.tolist(),
    )

    log.info("Successfully enrolled user '%s' directly from full continuous %.1fs audio clip", name, duration_s)
    return {
        "speaker": name,
        "samples_count": 1,
        "duration_sec": round(duration_s, 1),
        "has_voiceprint": True,
    }


def identify_embedding(test_emb: torch.Tensor, threshold: float = 0.50) -> dict:
    """Compare a precomputed 192-d embedding against all enrolled user centroids."""
    active_profiles = {k: v["centroid"] for k, v in _profiles.items() if v["centroid"] is not None}

    if not active_profiles:
        return {
            "winner": None,
            "top_match": None,
            "top_score": 0.0,
            "confidence_pct": 0.0,
            "scores": {},
            "threshold": threshold,
        }

    scores = {}
    for name, centroid in active_profiles.items():
        sim = torch.dot(test_emb, centroid).item()
        scores[name] = round(max(0.0, sim), 4)

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    conf_pct = round(min(99.9, max(1.0, (best_score - 0.2) / 0.60 * 100)), 1) if best_score > 0.20 else round(max(1.0, best_score * 50), 1)
    winner = best_name if best_score >= threshold else None

    return {
        "winner": winner,
        "top_match": best_name,
        "top_score": best_score,
        "confidence_pct": conf_pct,
        "scores": scores,
        "threshold": threshold,
    }


def identify_speaker(pcm_float32_16k: np.ndarray, threshold: float = 0.50) -> dict:
    """Compare a voice slice against all enrolled user centroids."""
    test_emb = extract_embedding(pcm_float32_16k)
    return identify_embedding(test_emb, threshold)


def get_users_detailed() -> List[dict]:
    """Return all user profiles with sample counts, voiceprints, and avatar URLs."""
    result = []
    for name, data in _profiles.items():
        result.append({
            "name": name,
            "samples_count": len(data["embeddings"]),
            "has_voiceprint": data["centroid"] is not None,
            "avatar_b64": data.get("avatar_b64"),
            "company_name": data.get("company_name", ""),
        })
    return result


def get_users() -> List[str]:
    """Return list of user names."""
    return list(_profiles.keys())


def reset_voice_model_centroids():
    """Reset all learned voice model centroids back to clean state, preserving user profiles, names, avatars, and companies."""
    for name, data in _profiles.items():
        data["embeddings"] = []
        data["centroid"] = None
        data["enrolled_centroid"] = None
        db.upsert_profile(
            name=name,
            avatar_b64=data.get("avatar_b64"),
            company_name=data.get("company_name"),
            samples_count=0,
            centroid_list=None,
        )
    log.info("Reset voice model centroids for all profiles (profiles preserved).")
    return {"status": "voice_model_reset", "users_count": len(_profiles)}


def clear_session():
    """Clear all user profiles and voiceprints."""
    _profiles.clear()
    db.clear_all_profiles()
    log.info("All user profiles & voiceprints cleared.")
    return {"status": "cleared"}
