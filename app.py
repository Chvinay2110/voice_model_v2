"""
app.py

DEEBUG live transcription server with AssemblyAI streaming tokens,
SpeechBrain ECAPA-TDNN voiceprint learning, persistent SQLite database,
Server-Sent Events (SSE) for instant admin synchronization, and Gemini AI analysis.
"""

import base64
import json
import logging
import os
import queue
import traceback
from collections import Counter

import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_from_directory

import db
import speaker_id_engine
from assemblyai_engine import create_temporary_token

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Gemini API key for AI analysis & Word Cloud
GEMINI_API_KEY = "AQ.Ab8RN6JhC0axQXrkr8hXPXcc68PT--FFO_8Srqk0blRjepHMFg"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__, static_folder=None)

# ── Initialize SQLite & In-Memory Store ──────────────────────────────────────
turn_store = {}
_sse_subscribers = []


def broadcast_sse(event_type: str, data: dict = None):
    """Push real-time instant notification to all connected admin panels."""
    payload = f"event: {event_type}\ndata: {json.dumps(data or {})}\n\n"
    for q in list(_sse_subscribers):
        try:
            q.put_nowait(payload)
        except Exception:
            if q in _sse_subscribers:
                _sse_subscribers.remove(q)


def init_app_state():
    """Load profiles and recent turns from SQLite into memory on startup."""
    speaker_id_engine.init_profiles_from_db()
    db_turns = db.get_all_turns()
    for t in db_turns:
        to = t["turn_order"]
        emb = None
        if t.get("audio_b64"):
            try:
                pcm = _decode_pcm_base64(t["audio_b64"])
                if len(pcm) >= int(0.3 * 16000):
                    emb = speaker_id_engine.extract_embedding(pcm)
            except Exception:
                pass

        turn_store[to] = {
            "audio_b64": t.get("audio_b64"),
            "embedding": emb,
            "text": t.get("text", ""),
            "speaker_label": t.get("speaker_label", "A"),
            "predicted_speaker": t.get("predicted_speaker"),
            "predicted_confidence": t.get("predicted_confidence", 0.0),
            "scores": {},
            "tagged_as": t.get("tagged_as"),
        }
    log.info("Initialized app with %d turns from database.", len(turn_store))


init_app_state()


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


@app.route("/api/stream-events")
def api_stream_events():
    """SSE endpoint for instant sub-millisecond updates to the admin panel."""

    def event_stream():
        q = queue.Queue(maxsize=200)
        _sse_subscribers.append(q)
        try:
            yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"
            while True:
                msg = q.get()
                yield msg
        except GeneratorExit:
            if q in _sse_subscribers:
                _sse_subscribers.remove(q)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/streaming-token", methods=["GET"])
def api_streaming_token():
    """Mints a short-lived AssemblyAI streaming token server-side."""
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
    "sort",
    "right",
    "gone",
    "young",
    "youngest",
    "higher",
    "country",
    "kilometers",
    "colleges",
    "could",
    "class",
    "india",
    "lit",
    "dropped",
    "becoming",
    "village",
    "something",
    "sometimes",
    "under",
    "been",
    "people",
    "think",
    "about",
    "which",
    "their",
    "there",
    "would",
    "these",
    "those",
    "where",
    "every",
    "other",
    "after",
    "first",
    "great",
    "going",
    "doing",
    "having",
    "saying",
    "really",
    "always",
    "never",
    "point",
    "thing",
    "things",
    "words",
    "speak",
    "speaking",
}


def extract_local_wordcloud(transcript):
    """Fallback local n-gram extraction for meaningful phrases."""
    lines = [l.split(":", 1)[-1].strip() for l in transcript.split("\n") if ":" in l or l.strip()]
    raw_text = " ".join(lines).lower()

    stopwords = {
        "the",
        "and",
        "is",
        "in",
        "to",
        "of",
        "that",
        "it",
        "with",
        "as",
        "for",
        "was",
        "on",
        "are",
        "by",
        "this",
        "be",
        "at",
        "from",
        "or",
        "an",
        "they",
        "we",
        "you",
        "i",
        "he",
        "she",
        "me",
        "my",
        "have",
        "has",
        "had",
        "not",
        "but",
        "what",
        "all",
        "were",
        "when",
        "can",
        "your",
        "said",
        "there",
        "use",
        "each",
        "which",
        "she",
        "do",
        "how",
        "their",
        "if",
        "will",
        "up",
        "other",
        "about",
        "out",
        "many",
        "then",
        "them",
        "these",
        "so",
        "some",
        "her",
        "would",
        "make",
        "like",
        "him",
        "into",
        "time",
        "has",
        "look",
        "two",
        "more",
        "write",
        "go",
        "see",
        "number",
        "no",
        "way",
        "could",
        "people",
        "my",
        "than",
        "first",
        "water",
        "been",
        "call",
        "who",
        "oil",
        "its",
        "now",
        "find",
        "long",
        "down",
        "day",
        "did",
        "get",
        "come",
        "made",
        "may",
        "part",
    }

    words = [w.strip('.,!?"():;') for w in raw_text.split() if w.strip('.,!?"():;')]
    phrases = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if (
            len(w1) > 2
            and len(w2) > 2
            and w1 not in stopwords
            and w2 not in stopwords
            and w1 not in GENERIC_BAN_WORDS
            and w2 not in GENERIC_BAN_WORDS
        ):
            phrases.append(f"{w1.capitalize()} {w2.capitalize()}")

    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i + 1], words[i + 2]
        if (
            len(w1) > 2
            and len(w3) > 2
            and w1 not in stopwords
            and w3 not in stopwords
            and w1 not in GENERIC_BAN_WORDS
            and w3 not in GENERIC_BAN_WORDS
        ):
            phrases.append(f"{w1.capitalize()} {w2} {w3.capitalize()}")

    counts = Counter(phrases).most_common(10)
    if not counts:
        return []

    weights = [48, 38, 32, 26, 22, 18, 15, 14, 13, 12]
    return [[p, weights[min(idx, len(weights) - 1)]] for idx, (p, _) in enumerate(counts)]


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
        "1. Every single entry MUST be a 2 to 4 word domain topic phrase (e.g. 'Higher Education Reforms', 'Youth"
        " Population Growth', 'University Seat Capacity', 'Government Policy', 'Institutional Rankings', 'Employment"
        " Opportunities').\n"
        "2. STRICTLY FORBIDDEN: Do NOT output isolated single words, adverbs, or generic adjectives (e.g."
        " 'Exceptional', 'Fast', 'Simply', 'Pointing', 'Fact', 'Result', 'Booming', 'Managed', 'Number', 'Seats',"
        " 'Youth', 'Achievement', 'Increase').\n"
        "3. Assign steep importance weights between 15 and 50.\n\n"
        "Output ONLY a valid JSON array of objects with keys 'text' and 'weight'. Example:\n"
        '[{"text": "Higher Education Reforms", "weight": 48}, {"text": "Government Policy Debate", "weight": 36}]\n\n'
        f"Transcript:\n{transcript}"
    )

    raw_json = call_gemini(prompt, response_json=True, timeout=15)
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            words = []
            if isinstance(parsed, list):
                words = [
                    [item.get("text", "").strip(), int(item.get("weight", 20))]
                    for item in parsed
                    if item.get("text") and len(item.get("text").split()) >= 2
                ]
            elif isinstance(parsed, dict) and "words" in parsed:
                words = [
                    [item.get("text", "").strip(), int(item.get("weight", 20))]
                    for item in parsed["words"]
                    if item.get("text") and len(item.get("text").split()) >= 2
                ]

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


# ── Speaker ID & Turn Endpoints ──────────────────────────────────────────────


def _decode_pcm_base64(b64_str):
    raw_bytes = base64.b64decode(b64_str)
    pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0


@app.route("/api/identify-turn", methods=["POST"])
def api_identify_turn():
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
    detailed = speaker_id_engine.get_users_detailed()
    names = [u["name"] for u in detailed]
    return jsonify({"users": detailed, "names": names})


@app.route("/api/create-user", methods=["POST"])
def api_create_user():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    avatar_b64 = data.get("avatar_b64")
    if not name:
        return jsonify({"error": "User name is required."}), 400
    try:
        res = speaker_id_engine.create_user(name, avatar_b64=avatar_b64)
        broadcast_sse("users_updated", res)
        return jsonify(res)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/delete-user", methods=["POST"])
def api_delete_user():
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
        broadcast_sse("users_updated", {"action": "deleted", "name": name})
        return jsonify(res)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/upload-avatar", methods=["POST"])
def api_upload_avatar():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    avatar_b64 = data.get("avatar_b64", "")
    if not name or not avatar_b64:
        return jsonify({"error": "name and avatar_b64 required."}), 400
    try:
        res = speaker_id_engine.set_user_avatar(name, avatar_b64)
        broadcast_sse("users_updated", res)
        return jsonify(res)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/clear-voiceprints", methods=["POST"])
def api_clear_voiceprints():
    res = speaker_id_engine.clear_session()
    for entry in turn_store.values():
        entry["tagged_as"] = None
        entry["predicted_speaker"] = None
        entry["predicted_confidence"] = 0.0
        entry["scores"] = {}
    broadcast_sse("cleared", {})
    return jsonify(res)


@app.route("/api/clear-all", methods=["POST"])
def api_clear_all():
    turn_store.clear()
    speaker_id_engine.clear_session()
    db.clear_all_profiles()
    broadcast_sse("cleared", {})
    return jsonify({"status": "cleared", "turns_count": 0, "users_count": 0})


def _rebuild_all_centroids():
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

    tagged_as = predicted_speaker if (predicted_speaker and predicted_confidence >= 50.0) else None

    # Write to SQLite database
    db.upsert_turn(
        turn_order=turn_order,
        text=text,
        speaker_label=speaker_label,
        audio_b64=audio_b64,
        tagged_as=tagged_as,
        predicted_speaker=predicted_speaker,
        predicted_confidence=predicted_confidence,
    )

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

    if len(turn_store) > 150:
        oldest = sorted(turn_store.keys())[: len(turn_store) - 150]
        for k in oldest:
            del turn_store[k]

    if tagged_as and emb is not None:
        _rebuild_all_centroids()

    # Instant Real-time Push to Admin Panel
    broadcast_sse("turn_update", {"turn_order": turn_order, "text": text, "tagged_as": tagged_as})

    return jsonify({
        "status": "stored",
        "turn_order": turn_order,
        "predicted_speaker": predicted_speaker,
        "predicted_confidence": predicted_confidence,
        "scores": scores,
    })


@app.route("/api/turns", methods=["GET"])
def api_turns():
    """Returns stored turns directly from persistent SQLite database."""
    turns = db.get_all_turns()
    result = []
    for t in turns:
        to = t["turn_order"]
        entry = turn_store.get(to, {})
        result.append({
            "turn_order": to,
            "text": t["text"],
            "speaker_label": t["speaker_label"],
            "tagged_as": t["tagged_as"],
            "predicted_speaker": t["predicted_speaker"],
            "predicted_confidence": t.get("predicted_confidence", 0.0),
            "scores": entry.get("scores", {}),
            "has_audio": bool(t.get("audio_b64")),
        })
    return jsonify({"turns": result})


@app.route("/api/tag-turn-from-admin", methods=["POST"])
def api_tag_turn_from_admin():
    data = request.get_json(force=True) or {}
    turn_order = data.get("turn_order")
    speaker_name = (data.get("speaker_name") or "").strip()
    if turn_order is None or not speaker_name:
        return jsonify({"error": "turn_order and speaker_name required"}), 400

    turn_order = int(turn_order)
    entry = turn_store.get(turn_order)
    if not entry:
        db_turn = db.get_turn(turn_order)
        if db_turn:
            entry = {
                "audio_b64": db_turn.get("audio_b64"),
                "embedding": None,
                "text": db_turn.get("text", ""),
                "speaker_label": db_turn.get("speaker_label", "A"),
                "tagged_as": None,
            }
            turn_store[turn_order] = entry

    if not entry or not entry.get("audio_b64"):
        return jsonify({"error": f"No audio stored for turn {turn_order}"}), 404

    try:
        if entry.get("embedding") is None:
            pcm = _decode_pcm_base64(entry["audio_b64"])
            if len(pcm) >= int(0.3 * 16000):
                entry["embedding"] = speaker_id_engine.extract_embedding(pcm)

        entry["tagged_as"] = speaker_name
        db.update_turn_tag(turn_order, speaker_name)
        _rebuild_all_centroids()

        # Instant SSE broadcast
        broadcast_sse("tag_updated", {"turn_order": turn_order, "tagged_as": speaker_name})

        return jsonify({
            "status": "tagged",
            "turn_order": turn_order,
            "speaker_name": speaker_name,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
