"""
app.py

Minimal Flask backend for the Live Transcript app.

  GET  /                    -> index.html
  GET  /api/streaming-token -> mint a short-lived AssemblyAI token so the
                               browser can open its own real-time transcription
                               WebSocket directly (no audio proxied through Flask)

Speaker diarization is handled entirely by AssemblyAI's built-in
`speaker_labels=true` parameter on the Universal-3 Pro streaming model —
no local speaker-embedding pipeline needed.

Run with:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000 in Chrome or Edge (mic permission required).
"""

import base64
import logging
import os
import traceback

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

from assemblyai_engine import create_temporary_token
import speaker_id_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Gemini API key for AI analysis & Word Cloud
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__, static_folder=None)

# ── In-memory turn audio store (populated by main page, consumed by admin page) ──
# turn_store[turn_order] = {"audio_b64": str, "text": str, "speaker_label": str, "tagged_as": str|None}
turn_store = {}


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/admin")
@app.route("/admin/")
def admin_page():
    return send_from_directory(BASE_DIR, "admin.html")


@app.route("/api/streaming-token", methods=["GET"])
def api_streaming_token():
    """Mints a short-lived AssemblyAI streaming token server-side so the
    browser can open its real-time transcription WebSocket directly against
    AssemblyAI without the real ASSEMBLYAI_API_KEY ever reaching the client.
    """
    try:
        token_data = create_temporary_token()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify(token_data)
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3.6-flash"]

def call_gemini(prompt, response_json=False, timeout=15):
    """Executes prompt on Gemini Flash with automatic model fallback."""
    if not GEMINI_API_KEY:
        return None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        if response_json:
            body["generationConfig"] = {"response_mime_type": "application/json"}
        try:
            resp = requests.post(url, params={"key": GEMINI_API_KEY}, json=body, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            continue
    return None


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """Calls Gemini with the transcript and returns an analysis."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not set."}), 500
    data = request.get_json(force=True)
    transcript = (data or {}).get("transcript", "").strip()
    if not transcript:
        return jsonify({"error": "No transcript provided."}), 400

    prompt = (
        "You are an expert meeting analyst. Analyse this live transcript and provide:\n"
        "1. A concise summary (2-3 sentences)\n"
        "2. Key discussion points (bullet list)\n"
        "3. Any action items or follow-ups mentioned\n"
        "4. Overall tone / sentiment\n\n"
        f"Transcript:\n{transcript}"
    )

    result = call_gemini(prompt, timeout=25)
    if result:
        return jsonify({"result": result})
    return jsonify({"error": "Gemini API unavailable."}), 500


GENERIC_BAN_WORDS = {
    'sort', 'right', 'gone', 'young', 'youngest', 'higher', 'country', 'kilometers', 'colleges', 'could',
    'class', 'india', 'lit', 'dropped', 'becoming', 'village', 'something', 'sometimes', 'under', 'been',
    'works', 'how', 'etc', 'actually', 'mean', 'means', 'does', 'did', 'done', 'doing', 'managed',
    'happened', 'hours', 'specific', 'leak', 'late', 'let', 'ask', 'telling', 'told', 'pointing',
    'wrong', 'early', 'appropriate', 'understand', 'answer', 'answered', 'days', 'call', 'booming',
    'really', 'maybe', 'thing', 'things', 'part', 'parts', 'much', 'many', 'good', 'bad', 'fact',
    'like', 'okay', 'great', 'small', 'big', 'going', 'talk', 'discuss', 'point', 'need', 'simply',
    'take', 'give', 'make', 'know', 'see', 'think', 'want', 'said', 'say', 'tell', 'well', 'result',
    'seats', 'increase', 'achievement', 'population', 'numbers', 'number', 'youth', 'people', 'also'
}


def extract_local_wordcloud(transcript):
    import re
    from collections import Counter
    stopwords = {
        'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'with', 'about', 'into', 'through', 'during', 'before', 'after', 'from', 'up', 'down',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'what', 'which',
        'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'hello',
        'yeah', 'uh', 'um', 'okay', 'ok', 'like', 'please', 'speaker', 'going', 'talk', 'think',
        'know', 'see', 'well', 'say', 'said', 'tell', 'want', 'come', 'good', 'also', 'able',
        'under', 'been', 'fact', 'simply', 'managed', 'booming', 'youngest', 'result', 'numbers',
        'pointing', 'exceptional', 'fast', 'seats', 'increase', 'achievement', 'population', 'number',
        'sort', 'right', 'gone', 'young', 'higher', 'country', 'kilometers', 'could', 'lit',
        'dropped', 'becoming', 'village', 'something', 'sometimes', 'works', 'actually', 'mean'
    }
    clean = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|Speaker\s+[A-Z\?]:?', ' ', transcript, flags=re.IGNORECASE)
    raw_tokens = re.findall(r'[a-zA-Z\u0900-\u097F]{3,}', clean)
    meaningful = [w.capitalize() for w in raw_tokens if w.lower() not in stopwords]
    
    # Generate 2-word topic phrases
    phrases = []
    for i in range(len(meaningful) - 1):
        w1, w2 = meaningful[i], meaningful[i+1]
        if w1.lower() != w2.lower():
            phrases.append(f"{w1} {w2}")
            
    counts = Counter(phrases).most_common(10)
    if not counts:
        return []
    
    weights = [48, 38, 32, 26, 22, 18, 15, 14, 13, 12]
    return [[p, weights[min(idx, len(weights)-1)]] for idx, (p, _) in enumerate(counts)]


@app.route("/api/wordcloud", methods=["POST"])
def api_wordcloud():
    """Extracts strictly multi-word substantive domain topics & macro-concepts."""
    data = request.get_json(force=True) or {}
    transcript = data.get("transcript", "").strip()
    if not transcript:
        return jsonify({"words": []})

    prompt = (
        "You are an executive discourse analyst.\n"
        "Analyze this conversation transcript and extract 8 to 12 CORE MACRO-TOPICS and conceptual subjects.\n\n"
        "MANDATORY REQUIREMENTS:\n"
        "1. Every single entry MUST be a 2 to 4 word domain topic phrase (e.g. 'Higher Education Reforms', 'Youth Population Growth', 'University Seat Capacity', 'Government Policy', 'Institutional Rankings', 'Employment Opportunities').\n"
        "2. STRICTLY FORBIDDEN: Do NOT output isolated single words, adverbs, or generic adjectives (e.g. 'Exceptional', 'Fast', 'Simply', 'Pointing', 'Fact', 'Result', 'Booming', 'Managed', 'Number', 'Seats', 'Youth', 'Achievement', 'Increase').\n"
        "3. Assign steep importance weights between 15 and 50.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return ONLY JSON in this exact structure:\n"
        '{"words": [["Higher Education Reforms", 50], ["Youth Population Growth", 40], ["University Seat Capacity", 34], ["Institutional Rankings", 28], ["Government Accountability", 22]]}'
    )

    raw_json = call_gemini(prompt, response_json=True, timeout=14)
    if raw_json:
        try:
            import json
            parsed = json.loads(raw_json)
            words = parsed.get("words", [])
            # Enforce 2+ word phrases and filter out generic words
            clean_words = []
            for w in words:
                if isinstance(w, (list, tuple)) and len(w) >= 2:
                    phrase = str(w[0]).strip()
                    tokens = phrase.split()
                    if len(tokens) >= 2 and not any(t.lower() in GENERIC_BAN_WORDS for t in tokens):
                        clean_words.append([phrase, int(w[1])])
            if len(clean_words) >= 3:
                return jsonify({"words": clean_words})
        except Exception:
            pass

    local_list = extract_local_wordcloud(transcript)
    return jsonify({"words": local_list})


# ── Speaker ID / Admin Tagging Endpoints ──────────────────────────────────────

def _decode_pcm_base64(b64_str):
    """Decode a base64-encoded 16-bit signed little-endian PCM buffer
    (16 kHz, mono) into a float32 numpy array in [-1, 1]."""
    raw_bytes = base64.b64decode(b64_str)
    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0


@app.route("/api/tag-turn-speaker", methods=["POST"])
def api_tag_turn_speaker():
    """Admin tags a turn with a speaker name.  The turn's audio slice is
    used to register (or update) that speaker's voiceprint.

    Request JSON:
      {
        "turn_order": 3,
        "speaker_name": "Vinay",
        "audio_b64": "<base64 encoded 16-bit PCM, 16kHz mono>"
      }
    """
    data = request.get_json(force=True) or {}
    speaker_name = (data.get("speaker_name") or "").strip()
    audio_b64 = data.get("audio_b64", "")

    if not speaker_name:
        return jsonify({"error": "speaker_name is required."}), 400
    if not audio_b64:
        return jsonify({"error": "audio_b64 is required."}), 400

    try:
        pcm = _decode_pcm_base64(audio_b64)
        if len(pcm) < int(0.3 * 16000):  # need at least 0.3s
            return jsonify({"error": "Audio slice too short (need ≥0.3s)."}), 400
        result = speaker_id_engine.register_voiceprint(speaker_name, pcm)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/identify-turn", methods=["POST"])
def api_identify_turn():
    """Identify who spoke a given turn by matching its audio against all
    registered voiceprints.

    Request JSON:
      {
        "turn_order": 5,
        "audio_b64": "<base64 encoded 16-bit PCM, 16kHz mono>"
      }
    """
    data = request.get_json(force=True) or {}
    audio_b64 = data.get("audio_b64", "")

    if not audio_b64:
        return jsonify({"error": "audio_b64 is required."}), 400

    try:
        pcm = _decode_pcm_base64(audio_b64)
        if len(pcm) < int(0.3 * 16000):
            return jsonify({"error": "Audio slice too short."}), 400
        result = speaker_id_engine.identify_speaker(pcm)
        result["turn_order"] = data.get("turn_order")
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/registered-users", methods=["GET"])
def api_registered_users():
    """Return the list of currently registered speaker profiles."""
    detailed = speaker_id_engine.get_users_detailed()
    names = [u["name"] for u in detailed]
    return jsonify({
        "users": detailed,
        "names": names,
    })


@app.route("/api/create-user", methods=["POST"])
def api_create_user():
    """Create a new user profile so it immediately appears in dropdowns."""
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    avatar_b64 = data.get("avatar_b64")
    if not name:
        return jsonify({"error": "User name is required."}), 400
    try:
        res = speaker_id_engine.create_user(name, avatar_b64=avatar_b64)
        return jsonify(res)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/delete-user", methods=["POST"])
def api_delete_user():
    """Delete a user profile, remove from turns, and rebuild centroids."""
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "User name is required."}), 400
    try:
        res = speaker_id_engine.delete_user(name)
        for entry in list(turn_store.values()):
            if (entry.get("tagged_as") or "").lower() == name.lower():
                entry["tagged_as"] = None
        _rebuild_all_centroids()
        return jsonify(res)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/upload-avatar", methods=["POST"])
def api_upload_avatar():
    """Upload or update avatar picture for a user."""
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    avatar_b64 = data.get("avatar_b64", "")
    if not name or not avatar_b64:
        return jsonify({"error": "name and avatar_b64 required."}), 400
    try:
        res = speaker_id_engine.set_user_avatar(name, avatar_b64)
        return jsonify(res)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/clear-voiceprints", methods=["POST"])
def api_clear_voiceprints():
    """Clear all registered voiceprints (fresh session)."""
    res = speaker_id_engine.clear_session()
    for entry in turn_store.values():
        entry["tagged_as"] = None
        entry["predicted_speaker"] = None
        entry["predicted_confidence"] = 0.0
        entry["scores"] = {}
    return jsonify(res)


@app.route("/api/clear-all", methods=["POST"])
def api_clear_all():
    """Wipe all turns, audio buffers, registered users and voiceprints for a brand new session."""
    turn_store.clear()
    speaker_id_engine.clear_session()
    return jsonify({"status": "cleared", "turns_count": 0, "users_count": 0})


def _rebuild_all_centroids():
    """Aggregates all sentence embeddings assigned to each user and updates
    their ECAPA-TDNN centroids in speaker_id_engine."""
    grouped = {}
    for entry in list(turn_store.values()):
        speaker = (entry.get("tagged_as") or "").strip()
        emb = entry.get("embedding")
        if speaker and emb is not None:
            if speaker not in grouped:
                grouped[speaker] = []
            grouped[speaker].append(emb)
    speaker_id_engine.sync_all_user_centroids(grouped)
    return grouped


@app.route("/api/store-turn-audio", methods=["POST"])
def api_store_turn_audio():
    """Main page uploads finalized turn audio here. Extracts embedding,
    identifies speaker, and updates the user's centroid."""
    data = request.get_json(force=True) or {}
    turn_order = data.get("turn_order")
    if turn_order is None:
        return jsonify({"error": "turn_order required"}), 400

    turn_order = int(turn_order)
    audio_b64 = data.get("audio_b64", "")
    text = data.get("text", "")
    speaker_label = data.get("speaker_label", "A")

    emb = None
    predicted_speaker = None
    predicted_confidence = 0.0
    scores = {}

    # Extract embedding and run identification if audio is valid
    if audio_b64:
        try:
            pcm = _decode_pcm_base64(audio_b64)
            if len(pcm) >= int(0.3 * 16000):
                emb = speaker_id_engine.extract_embedding(pcm)
                id_res = speaker_id_engine.identify_embedding(emb)
                scores = id_res.get("scores", {})
                predicted_speaker = id_res.get("top_match") or id_res.get("winner")
                predicted_confidence = id_res.get("confidence_pct", 0.0)
        except Exception:
            pass

    # Default tag to predicted speaker if match is found
    tagged_as = predicted_speaker if (predicted_speaker and predicted_confidence >= 50.0) else None

    turn_store[turn_order] = {
        "audio_b64": audio_b64,
        "embedding": emb,
        "text": text,
        "speaker_label": speaker_label,
        "predicted_speaker": predicted_speaker,
        "predicted_confidence": predicted_confidence,
        "scores": scores,
        "tagged_as": tagged_as,
    }

    # Keep only last 100 turns
    if len(turn_store) > 100:
        oldest = sorted(turn_store.keys())[:len(turn_store) - 100]
        for k in oldest:
            del turn_store[k]

    # Rebuild all centroids with the new sentence included
    if tagged_as and emb is not None:
        _rebuild_all_centroids()

    return jsonify({
        "status": "stored",
        "turn_order": turn_order,
        "predicted_speaker": predicted_speaker,
        "predicted_confidence": predicted_confidence,
        "scores": scores,
    })


@app.route("/api/turns", methods=["GET"])
def api_turns():
    """Returns the list of stored turns with predictions and scores for admin page."""
    result = []
    for to in sorted(turn_store.keys(), reverse=True):
        entry = turn_store[to]
        result.append({
            "turn_order": to,
            "text": entry["text"],
            "speaker_label": entry["speaker_label"],
            "tagged_as": entry.get("tagged_as"),
            "predicted_speaker": entry.get("predicted_speaker"),
            "predicted_confidence": entry.get("predicted_confidence", 0.0),
            "scores": entry.get("scores", {}),
            "has_audio": bool(entry.get("audio_b64")),
        })
    return jsonify({"turns": result})


@app.route("/api/tag-turn-from-admin", methods=["POST"])
def api_tag_turn_from_admin():
    """Admin page tags or reassigns a turn to a speaker.

    Dynamically moves the turn's embedding signal to the new user and
    recomputes centroids across ALL assigned sentences for every user.
    """
    data = request.get_json(force=True) or {}
    turn_order = data.get("turn_order")
    speaker_name = (data.get("speaker_name") or "").strip()
    if turn_order is None or not speaker_name:
        return jsonify({"error": "turn_order and speaker_name required"}), 400

    turn_order = int(turn_order)
    entry = turn_store.get(turn_order)
    if not entry or not entry.get("audio_b64"):
        return jsonify({"error": f"No audio stored for turn {turn_order}"}), 404

    try:
        # If embedding not cached yet, compute it now
        if entry.get("embedding") is None:
            pcm = _decode_pcm_base64(entry["audio_b64"])
            if len(pcm) >= int(0.3 * 16000):
                entry["embedding"] = speaker_id_engine.extract_embedding(pcm)

        # Set new assignment
        entry["tagged_as"] = speaker_name

        # Rebuild all centroids across all users from all currently assigned sentences
        grouped = _rebuild_all_centroids()

        # Re-evaluate predictions on all other turns using the updated centroids
        for t_entry in list(turn_store.values()):
            t_emb = t_entry.get("embedding")
            if t_emb is not None:
                t_id = speaker_id_engine.identify_embedding(t_emb)
                t_entry["scores"] = t_id.get("scores", {})
                t_entry["predicted_speaker"] = t_id.get("top_match") or t_id.get("winner")
                t_entry["predicted_confidence"] = t_id.get("confidence_pct", 0.0)

        samples_count = len(grouped.get(speaker_name, []))
        return jsonify({
            "speaker": speaker_name,
            "status": "updated",
            "samples_count": samples_count,
            "embedding_dim": 192,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
