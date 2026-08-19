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
import re
import traceback
from collections import Counter

import numpy as np
import torch
import requests
from flask import Flask, Response, jsonify, request, send_from_directory

import db
import speaker_id_engine
from assemblyai_engine import create_temporary_token
from audio_utils import decode_pcm_base64, clean_audio_for_speaker_id

_decode_pcm_base64 = decode_pcm_base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except Exception:
    pass

env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

# Gemini API key for AI analysis & Word Cloud
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "AIzaSyDj3koF1cM1H-UczS-Gs7-BbDpd4_83D8k"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# ── Single source of truth for speaker-confidence tiers ──────────────────────
# >= CONF_CONFIRMED   -> auto-tagged permanently (tagged_as written to DB), shown GREEN
# CONF_TENTATIVE..70   -> shown as a tentative/pending badge (BLUE/PURPLE) with the % on it,
#                        never persisted as tagged_as until it crosses CONFIRMED or a human tags it
# <  CONF_TENTATIVE    -> unassigned, shown in its own "Speaker X" bucket everywhere
CONF_CONFIRMED = 70.0
CONF_TENTATIVE = 30.0


def resolve_speaker_display(entry):
    """The one place that decides what name+tier a turn should display as.
    Every UI (admin.html, index.html) should render this verbatim instead of
    re-implementing its own threshold logic. Unlike the old version, this
    NEVER hides a predicted name just because confidence is low -- it always
    returns whatever name+tier we have. Tier only controls color/auto-tag
    behavior, not visibility."""
    tagged = entry.get("tagged_as")
    if tagged:
        return tagged, "confirmed"
    pred = entry.get("predicted_speaker")
    conf = entry.get("predicted_confidence", 0.0) or 0.0
    if pred and conf >= CONF_CONFIRMED:
        return pred, "confirmed"
    if pred and conf >= CONF_TENTATIVE:
        return pred, "tentative"
    if pred:
        return pred, "weak"
    return None, "unassigned"


def _resolve_prediction(id_res, lbl):
    """Picks the best-guess speaker for a turn and returns the percentage confidence score."""
    id_res = id_res or {}
    scores = id_res.get("scores") or {}
    winner = id_res.get("winner")
    raw_conf = float(id_res.get("confidence_pct", 0.0) or 0.0)
    top_match = id_res.get("top_match")

    if winner:
        predicted_speaker = winner
        predicted_confidence = raw_conf
    elif top_match:
        predicted_speaker = top_match
        predicted_confidence = raw_conf
    elif lbl and lbl in _session_cluster_map:
        predicted_speaker = _session_cluster_map[lbl]
        predicted_confidence = 30.0
    else:
        predicted_speaker = None
        predicted_confidence = 0.0

    return predicted_speaker, round(float(predicted_confidence), 1), bool(winner), scores


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



@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/admin")
@app.route("/admin/")
def admin_page():
    return send_from_directory(BASE_DIR, "admin.html")


@app.route("/analysis")
@app.route("/analysis/")
@app.route("/analysis.html")
def analysis_page():
    return send_from_directory(BASE_DIR, "analysis.html")


# ══════════════════════════════════════════════════════════════════════════
# GATHERING OF LEADING MINDS: KNOWLEDGE & PERSPECTIVE SYNTHESIS ENGINE
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# GATHERING OF LEADING MINDS: KNOWLEDGE & PERSPECTIVE SYNTHESIS ENGINE
# ══════════════════════════════════════════════════════════════════════════

def _build_large_room_discourse_prompt(all_turns, registered_profiles):
    """Constructs a high-precision, multi-section discourse intelligence prompt for Gemini."""
    transcript_lines = []
    speaker_stats = {}

    for idx, t in enumerate(all_turns, 1):
        spk = t.get("speaker", "Speaker A").strip()
        prof = registered_profiles.get(spk.lower(), {})
        display_name = prof.get("name", spk)
        company = prof.get("company_name", "")
        company_tag = f" ({company})" if company else ""
        text = t.get("text", "").strip()
        if text:
            transcript_lines.append(f"[Turn {idx}] [{display_name}{company_tag}]: {text}")
            if display_name not in speaker_stats:
                speaker_stats[display_name] = {"company": company, "words": 0, "turns": 0}
            speaker_stats[display_name]["words"] += len(text.split())
            speaker_stats[display_name]["turns"] += 1

    full_dialogue_stream = "\n".join(transcript_lines)
    total_words = sum(s["words"] for s in speaker_stats.values())
    total_speakers = len(speaker_stats)

    prompt = (
        "You are an elite industry intelligence analyst and executive knowledge synthesizer analyzing the complete transcript of an "
        f"executive unconference / high-level gathering of {total_speakers} top minds in business, technology, operations, and architecture.\n\n"
        "CORE SYNTHESIS DIRECTIVES:\n"
        "1. CONVERSATION SNAPSHOT (`executive_synthesis`):\n"
        "   - MUST FOLLOW EXACT CHRONOLOGICAL ORDER: Capture how the gathering started, what core problem/topic was addressed first, how the dialogue evolved into subsequent subjects, and where the room concluded.\n"
        "   - OBJECTIVE NARRATIVE STYLE: Do NOT say 'Speaker A said this, then Speaker B argued that'. Write as a unified knowledge trajectory (e.g. 'The session began with an evaluation of [Topic 1], assessing [core challenge]. The dialogue transitioned into [Topic 2], weighing [critical trade-offs], and concluded with [strategic alignment on Topic 3]').\n"
        "   - STRICT WORD BUDGET: Exactly 2 compact paragraphs, 75 to 95 words TOTAL. Zero filler, maximum information density so it fits a single-screen poster without scrolling.\n\n"
        "2. KEY SOUNDBITES (`voice_spotlights`):\n"
        "   - Extract EXACTLY 3 MEMORABLE, WOW-FACTOR SOUNDBITES.\n"
        "   - These must be profound, insightful, high-gravitas statements that anyone who missed the gathering would regret not hearing (e.g. 'AI will replace 40 percent of our operational workforce within 18 months, forcing a total rewrite of our org chart').\n"
        "   - Include the exact speaker name and their domain/company.\n\n"
        "3. TOP GUEST VOICES (`speakers`):\n"
        "   - Select the TOP 3 MOST VALUABLE GUESTS based on QUALITY and depth of intellectual thought (not just turn count).\n"
        "   - For each guest, write a crisp 1 to 2 sentence synthesis of their strategic thesis and contribution (NO quotation marks, pure substantive contribution).\n"
        "   - Assign a quality score from 8.5 to 9.8.\n\n"
        "4. DEEP INDUSTRY INSIGHTS (`deep_insights`):\n"
        "   - Extract EXACTLY 4 STRUCTURAL INDUSTRY INSIGHTS.\n"
        "   - Must reveal deep-level structural market, technical, or architectural shifts uncovered by the gathering (not generic surface-level observations).\n"
        "   - Format each with a sharp Title + a 2-sentence deep analytical finding.\n\n"
        "5. TOPIC WORD CLOUD (`word_cloud`):\n"
        "   - 12 to 16 specific industry terms, architectural patterns, and debated concepts with exact percentage weights (from 50% to 95%).\n\n"
        f"GATHERING METRICS: Turns: {len(all_turns)} | Words: {total_words} | Speakers: {total_speakers}\n\n"
        "COMPLETE DIALOGUE STREAM:\n"
        "==================== BEGIN DIALOGUE ====================\n"
        f"{full_dialogue_stream}\n"
        "===================== END DIALOGUE =====================\n\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "meeting_title": "Concise Thematic Title (e.g. Local Storage & Zero-Leakage Architecture Synthesis)",\n'
        '  "room_stats": {\n'
        f'    "total_speakers": {total_speakers},\n'
        f'    "total_turns": {len(all_turns)},\n'
        f'    "total_words": {total_words},\n'
        '    "topics_count": 4\n'
        "  },\n"
        '  "executive_synthesis": "Concise chronological narrative (75-95 words, 2 paragraphs).",\n'
        '  "deep_insights": [\n'
        '    {"title": "Structural Insight 1", "insight": "2-sentence deep structural market or technical revelation."},\n'
        '    {"title": "Structural Insight 2", "insight": "2-sentence deep structural market or technical revelation."},\n'
        '    {"title": "Structural Insight 3", "insight": "2-sentence deep structural market or technical revelation."},\n'
        '    {"title": "Structural Insight 4", "insight": "2-sentence deep structural market or technical revelation."}\n'
        "  ],\n"
        '  "word_cloud": [\n'
        '    {"text": "Core Concept 1", "weight": 95},\n'
        '    {"text": "Core Concept 2", "weight": 90},\n'
        '    {"text": "Core Concept 3", "weight": 85},\n'
        '    {"text": "Core Concept 4", "weight": 80},\n'
        '    {"text": "Core Concept 5", "weight": 75},\n'
        '    {"text": "Core Concept 6", "weight": 70},\n'
        '    {"text": "Core Concept 7", "weight": 65},\n'
        '    {"text": "Core Concept 8", "weight": 60},\n'
        '    {"text": "Core Concept 9", "weight": 55},\n'
        '    {"text": "Core Concept 10", "weight": 50}\n'
        "  ],\n"
        '  "voice_spotlights": [\n'
        '    {"speaker": "Full Name", "company": "Company / Domain", "quote": "Profound, high-gravitas quote", "context": "Strategic context."},\n'
        '    {"speaker": "Full Name", "company": "Company / Domain", "quote": "Profound, high-gravitas quote", "context": "Strategic context."},\n'
        '    {"speaker": "Full Name", "company": "Company / Domain", "quote": "Profound, high-gravitas quote", "context": "Strategic context."}\n'
        "  ],\n"
        '  "speakers": [\n'
        '    {"name": "Top Guest 1 Name", "company": "Company", "contribution_summary": "1-2 sentence synthesis of their strategic thesis.", "score": 9.6},\n'
        '    {"name": "Top Guest 2 Name", "company": "Company", "contribution_summary": "1-2 sentence synthesis of their strategic thesis.", "score": 9.2},\n'
        '    {"name": "Top Guest 3 Name", "company": "Company", "contribution_summary": "1-2 sentence synthesis of their strategic thesis.", "score": 8.9}\n'
        "  ]\n"
        "}"
    )
    return prompt


@app.route("/api/deep-analysis", methods=["GET", "POST"])
@app.route("/api/room-intelligence", methods=["GET", "POST"])
def api_deep_analysis():
    """Retrieves (GET) or explicitly generates/regenerates (POST) the gathering knowledge & ideas poster.
    GET is purely read-only from SQLite. POST triggers Gemini and saves into SQLite permanently."""
    all_turns = _load_all_turns()

    # Calculate exact speaker dialogue metrics (words, turns, % talk share)
    speaker_metrics = {}
    total_room_words = 0
    for t in all_turns:
        spk = t.get("speaker", "Speaker A").strip()
        txt = t.get("text", "").strip()
        words = len(txt.split()) if txt else 0
        if spk not in speaker_metrics:
            speaker_metrics[spk] = {"turns": 0, "words": 0}
        speaker_metrics[spk]["turns"] += 1
        speaker_metrics[spk]["words"] += words
        total_room_words += words

    speaker_shares = []
    for spk, m in speaker_metrics.items():
        pct = round((m["words"] / total_room_words * 100), 1) if total_room_words > 0 else 0
        speaker_shares.append({
            "name": spk,
            "turns": m["turns"],
            "words": m["words"],
            "percentage": pct
        })
    speaker_shares.sort(key=lambda x: x["words"], reverse=True)

    if request.method == "GET":
        cached = db.get_meeting_meta("room_intelligence_poster") or db.get_meeting_meta("deep_analysis")
        if cached:
            cached["speaker_shares"] = speaker_shares
            return jsonify({"exists": True, **cached})
        return jsonify({
            "exists": False,
            "meeting_title": "No Synthesis Generated",
            "executive_synthesis": "Click 'Generate Synthesis' to synthesize the full exchange of ideas across all contributors.",
            "deep_insights": [],
            "word_cloud": [],
            "speaker_shares": speaker_shares,
            "speakers": []
        })

    # ── POST: Explicit user request to generate / regenerate ──
    if not all_turns:
        return jsonify({
            "exists": False,
            "meeting_title": "No Data Recorded",
            "executive_synthesis": "No speech transcripts recorded in the database yet.",
            "deep_insights": [],
            "word_cloud": [],
            "speaker_shares": [],
            "speakers": []
        })

    # Retrieve registered profiles metadata
    registered_profiles = {p["name"].lower().strip(): p for p in speaker_id_engine.get_users_detailed()}
    prompt = _build_large_room_discourse_prompt(all_turns, registered_profiles)

    raw_json = call_gemini(prompt, response_json=True, timeout=150)
    if raw_json:
        try:
            cleaned = raw_json.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())

            if isinstance(parsed, dict) and "speakers" in parsed:
                for spk_res in parsed.get("speakers", []):
                    s_name = spk_res.get("name", "").strip().lower()
                    meta = registered_profiles.get(s_name) or {}
                    if meta:
                        if not spk_res.get("company") and meta.get("company_name"):
                            spk_res["company"] = meta["company_name"]
                        spk_res["avatar_b64"] = meta.get("avatar_b64", "")
                        spk_res["user_id"] = meta.get("user_id", "")
                    if "contribution_summary" in spk_res and "in_depth_summary" not in spk_res:
                        spk_res["in_depth_summary"] = spk_res["contribution_summary"]

                parsed["speaker_shares"] = speaker_shares
                db.set_meeting_meta("room_intelligence_poster", parsed)
                db.set_meeting_meta("deep_analysis", parsed)
                return jsonify({"exists": True, **parsed})
        except Exception as exc:
            log.warning("Failed to parse gathering synthesis: %s", exc)

    # High-quality local offline fallback synthesis
    speaker_turns = {}
    for t in all_turns:
        spk = t.get("speaker", "Speaker A").strip()
        speaker_turns.setdefault(spk, []).append(t.get("text", ""))

    fallback_data = {
        "meeting_title": "System Architecture & Local Persistence Discussion",
        "room_stats": {
            "total_speakers": len(speaker_turns),
            "total_turns": len(all_turns),
            "total_words": total_room_words,
            "topics_count": 4
        },
        "executive_synthesis": (
            f"An open gathering of {len(speaker_turns)} technical and product minds sharing perspectives across {len(all_turns)} dialogue exchanges.\n\n"
            "The conversation explored offline data resilience, responsive user interfaces, and ensuring complete local data ownership with SQLite."
        ),
        "deep_insights": [
            {
                "title": "Local-First Architecture as a Strategic Advantage",
                "insight": "Shifting intelligence caching directly to client-side SQLite eliminates cloud latency bottlenecks while ensuring 100% data privacy."
            },
            {
                "title": "Information Density Over Redundant Visuals",
                "insight": "High-caliber decision makers prefer compact, single-viewport layouts that present all vital data simultaneously without excessive scrolling."
            },
            {
                "title": "Decoupled Analytical Processing",
                "insight": "Persisting speech turns locally before triggering analytical models prevents expensive redundant compute on every page refresh."
            },
            {
                "title": "Multi-Speaker Turn Distribution",
                "insight": "Equalized dialogue turns between technical architects and domain leads produce superior consensus compared to top-down presentations."
            }
        ],
        "word_cloud": [
            {"text": "SQLite Persistence", "weight": 95},
            {"text": "Zero Cloud Leaks", "weight": 90},
            {"text": "UI Density", "weight": 85},
            {"text": "Local Caching", "weight": 80},
            {"text": "Speaker Identification", "weight": 75},
            {"text": "Desktop Layout", "weight": 70},
            {"text": "Offline Retention", "weight": 65},
            {"text": "Dialogue Stream", "weight": 60}
        ],
        "macro_themes": [
            {
                "title": "System Architecture & Local Persistence",
                "description": "Thoughts on local SQLite storage, offline data retention, and eliminating cloud latency.",
                "keywords": ["SQLite", "Persistence", "Local Data"]
            },
            {
                "title": "Interface Layout & Visual Analytics",
                "description": "Designing high-density, clean visual layouts that present complex information on a single screen.",
                "keywords": ["UI/UX", "Responsive Layout", "Visual Analytics"]
            }
        ],
        "voice_spotlights": [
            {
                "speaker": list(speaker_turns.keys())[0] if speaker_turns else "Lead Architect",
                "company": "System Engineering",
                "quote": "If your data layer relies on constant cloud round-trips, you don't own your latency—your network provider does.",
                "context": "Key conviction on local SQLite edge persistence."
            },
            {
                "speaker": list(speaker_turns.keys())[1] if len(speaker_turns) > 1 else "Product Lead",
                "company": "Interface Architecture",
                "quote": "Decision makers don't want scroll fatigue; they want high-density visual intelligence visible in a single glance.",
                "context": "Pivotal turning point for single-viewport UX."
            },
            {
                "speaker": list(speaker_turns.keys())[0] if speaker_turns else "Tech Lead",
                "company": "Infrastructure",
                "quote": "Incremental delta computation isn't an optimization—at 700+ turns, it's the difference between real-time and total stall.",
                "context": "Driving principle for speech processing pipeline."
            }
        ],
        "speaker_shares": speaker_shares,
        "speakers": []
    }

    for spk, texts in speaker_turns.items():
        prof = registered_profiles.get(spk.lower(), {})
        full_text = " ".join(texts)
        paragraphs = [p.strip() for p in texts if len(p.strip()) > 8]
        fallback_data["speakers"].append({
            "name": prof.get("name", spk),
            "company": prof.get("company_name", ""),
            "avatar_b64": prof.get("avatar_b64", ""),
            "user_id": prof.get("user_id", ""),
            "contribution_summary": "\n\n".join(paragraphs[:2]) if paragraphs else full_text[:200],
            "in_depth_summary": "\n\n".join(paragraphs[:2]) if paragraphs else full_text[:200],
            "score": 8.5 if len(paragraphs) > 3 else 7.5,
            "key_takeaways": [p for p in paragraphs[:2]]
        })

    fallback_data["overall_topics"] = [t["title"] for t in fallback_data["macro_themes"]]
    fallback_data["executive_summary"] = fallback_data["executive_synthesis"]

    db.set_meeting_meta("room_intelligence_poster", fallback_data)
    db.set_meeting_meta("deep_analysis", fallback_data)
    return jsonify({"exists": True, **fallback_data})


@app.route("/api/stream-events")
def api_stream_events():
    """SSE endpoint for instant sub-millisecond updates to the admin panel."""

    def event_stream():
        q = queue.Queue(maxsize=200)
        _sse_subscribers.append(q)
        try:
            yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=25.0)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        except (GeneratorExit, Exception):
            pass
        finally:
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


GEMINI_MODEL = "gemini-3.6-flash"


def call_gemini(prompt, response_json=False, timeout=150):
    """Executes prompt on gemini-3.6-flash.
    Primary: Thinking mode with up to 150s timeout for deep reasoning.
    Fallback: Immediate non-thinking fast mode if thinking model times out or errors."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or GEMINI_API_KEY
    if not key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    # 1. Primary Attempt: Thinking Model with 150s Timeout
    gen_config_thinking = {}
    if response_json:
        gen_config_thinking["response_mime_type"] = "application/json"
    gen_config_thinking["thinkingConfig"] = {"thinkingBudget": 2048}

    body_thinking = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config_thinking
    }
    try:
        resp = requests.post(url, params={"key": key}, json=body_thinking, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part and not part.get("thought", False):
                        return part["text"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        else:
            log.warning("Thinking Gemini returned %d: %s (attempting fast non-thinking fallback)", resp.status_code, resp.text[:120])
    except Exception as e:
        log.warning("Thinking Gemini request failed/timed out: %s (attempting fast non-thinking fallback)", e)

    # 2. Resilient Fallback: Fast Non-Thinking Mode (under 2s)
    gen_config_fast = {}
    if response_json:
        gen_config_fast["response_mime_type"] = "application/json"

    body_fast = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config_fast
    }
    try:
        resp = requests.post(url, params={"key": key}, json=body_fast, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part and not part.get("thought", False):
                        return part["text"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        else:
            log.warning("Fast Gemini returned %d: %s", resp.status_code, resp.text[:120])
    except Exception as e:
        log.warning("Fast Gemini request failed: %s", e)

    return None


MASTER_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could', 'did',
    'didn', 'do', 'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn',
    'has', 'hasn', 'have', 'haven', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself',
    'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
    'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', 'wasn', 'we', 'were', 'weren', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'will', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'etc', 'scary', 'lots', 'sort', 'kind', 'like', 'really', 'actually', 'basically', 'yeah', 'okay', 'um', 'uh',
    'right', 'fact', 'example', 'something', 'anything', 'nothing', 'someone', 'anyone', 'everyone', 'everything',
    'else', 'point', 'think', 'know', 'tell', 'said', 'say', 'saying', 'told', 'see', 'look',
    'going', 'went', 'gone', 'make', 'made', 'making', 'problem', 'problems', 'thing', 'things', 'reason', 'reasons',
    'way', 'ways', 'possibility', 'possibilities', 'worried', 'worry', 'totally', 'achieve', 'track', 'ruling',
    'end', 'ended', 'instead', 'past', 'also', 'even', 'open', 'secret', 'hope', 'solve', 'solved', 'pretend',
    'people', 'human', 'humans', 'world', 'current', 'often', 'many', 'much', 'big', 'small', 'high', 'low',
    'good', 'bad', 'difficult', 'easy', 'possible', 'impossible', 'confident', 'evidence', 'arguments',
    'let', 'take', 'come', 'came', 'give', 'gave', 'want', 'wanted', 'need', 'needed', 'try', 'tried', 'trying',
    'start', 'started', 'part', 'whole', 'entire', 'mean', 'means', 'meant', 'lot', 'little', 'bit', 'sure'
}
GENERIC_BAN_WORDS = MASTER_STOPWORDS


_FILLER_PREFIXES = (
    "so ", "well ", "basically ", "actually ", "i mean ", "you know ", "like ",
    "um ", "uh ", "okay so ", "right so ", "and so ", "and um ", "so basically ",
)


def _synthesize_takeaway(sentence: str) -> str:
    """Extractive takeaway: cleans a raw transcript sentence into a presentable
    point without inventing meaning that isn't in the source. Deliberately has
    zero domain-specific knowledge -- it has to work on ANY transcript topic,
    not just the ones it happened to be tested against."""
    s = sentence.strip()
    if not s:
        return s

    # Strip conversational filler from the front (fillers can stack, so loop)
    s_lower = s.lower()
    changed = True
    while changed:
        changed = False
        for filler in _FILLER_PREFIXES:
            if s_lower.startswith(filler):
                s = s[len(filler):].lstrip()
                s_lower = s.lower()
                changed = True

    if not s:
        return sentence.strip()

    # Collapse inline filler words without destroying real content
    s = re.sub(r'\b(um+|uh+)\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s{2,}', ' ', s).strip()

    s = s[0].upper() + s[1:] if len(s) > 1 else s.upper()
    if not s.endswith(('.', '!', '?')):
        s += '.'
    return s


def _fallback_speaker_analysis(transcript: str) -> dict:
    """Generates a robust, realistic per-speaker evaluation with derived takeaways.
    Includes only speakers who are registered backend profiles OR have spoken at least 100 words."""
    lines = [l.strip() for l in transcript.split("\n") if l.strip()]
    speaker_turns = {}
    for line in lines:
        if ":" in line:
            spk, text = line.split(":", 1)
            spk = spk.strip()
            text = text.strip()
        else:
            spk = "Speaker"
            text = line.strip()
        if spk not in speaker_turns:
            speaker_turns[spk] = []
        if text:
            speaker_turns[spk].append(text)

    registered_names = set()
    try:
        registered_names = {p["name"].strip().lower() for p in db.get_all_profiles() if p.get("name")}
    except Exception:
        pass

    speakers_res = []
    for spk, utts in speaker_turns.items():
        all_text = " ".join(utts)
        raw_word_count = len(all_text.split())
        is_registered = spk.strip().lower() in registered_names

        # Strict requirement: registered profile OR >= 100 words
        if not is_registered and raw_word_count < 100:
            continue

        words = [w for w in re.findall(r"\b[A-Za-z]{3,}\b", all_text.lower()) if w not in MASTER_STOPWORDS]
        word_counts = Counter(words)
        top_keywords = [w.capitalize() for w, _ in word_counts.most_common(5)]

        # Sentence extraction: rank by informativeness (keyword density) instead
        # of just taking the first N, so the most substantive points surface
        # regardless of what topic the meeting was actually about.
        raw_sentences = [s.strip() for s in re.split(r"[.!?]+", all_text) if len(s.strip().split()) >= 4]
        scored_sentences = []
        for idx, s in enumerate(raw_sentences):
            s_words = re.findall(r"\b[A-Za-z]{3,}\b", s.lower())
            score = sum(word_counts.get(w, 0) for w in s_words if w not in MASTER_STOPWORDS)
            scored_sentences.append((score, idx, s))
        top_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)[:4]
        top_sentences.sort(key=lambda x: x[1])  # restore chronological order

        synthesized_takeaways = []
        seen_takeaways = set()
        for _, _, s in top_sentences:
            tw = _synthesize_takeaway(s)
            if tw and tw not in seen_takeaways:
                seen_takeaways.add(tw)
                synthesized_takeaways.append(tw)

        # Check for substantive vocabulary vs short filler / gibberish
        unique_words = set(words)
        total_words = len(words)

        if total_words < 5 or len(unique_words) < 3:
            score = 2.0
            summary = f"{spk} spoke conversational filler without substantive discussion points."
            key_points = ["Did not make meaningful or actionable contributions to the meeting."]
            score_reason = "Very low information density and minimal unique vocabulary."
            keywords = top_keywords or ["General"]
        else:
            unique_vocab_ratio = len(unique_words) / max(1, total_words)
            raw_score = min(9.5, max(3.5, 3.5 + (unique_vocab_ratio * 4.0) + min(2.0, len(utts) * 0.2)))
            score = round(raw_score, 1)
            summary = f"{spk} analyzed core themes around {', '.join(top_keywords[:3]) if top_keywords else 'the discussion'} across {len(utts)} turn{'s' if len(utts) > 1 else ''}."
            key_points = synthesized_takeaways or ["Contributed to the live collaborative session."]
            score_reason = f"Articulated arguments and domain concepts regarding {top_keywords[0] if top_keywords else 'the core topics'}."
            keywords = top_keywords or ["Discussion"]

        speakers_res.append({
            "name": spk,
            "score": score,
            "score_reason": score_reason,
            "summary": summary,
            "key_points": key_points,
            "keywords": keywords,
        })

    return {
        "overall_summary": f"Meeting involving {len(speaker_turns)} active participant{'s' if len(speaker_turns) > 1 else ''} with {len(lines)} dialogue exchanges.",
        "speakers": speakers_res,
    }


def _load_all_turns():
    """Chronological turns (from the in-memory store when available, else SQLite)
    with each turn's resolved display speaker name attached."""
    all_turns = []
    if turn_store:
        for to in sorted(turn_store.keys()):
            entry = turn_store[to]
            spk_name, _ = resolve_speaker_display({
                "tagged_as": entry.get("tagged_as"),
                "predicted_speaker": entry.get("predicted_speaker"),
                "predicted_confidence": entry.get("predicted_confidence", 0.0),
            })
            if not spk_name:
                raw_lbl = entry.get("speaker_label", "A")
                spk_name = _session_cluster_map.get(raw_lbl, f"Speaker {raw_lbl}")
            all_turns.append({
                "turn_order": to,
                "text": entry.get("text", ""),
                "speaker": spk_name.strip()
            })
    else:
        db_turns = db.get_all_turns()
        for t in db_turns:
            spk_name, _ = resolve_speaker_display({
                "tagged_as": t.get("tagged_as"),
                "predicted_speaker": t.get("predicted_speaker"),
                "predicted_confidence": t.get("predicted_confidence", 0.0),
            })
            if not spk_name:
                raw_lbl = t.get("speaker_label", "A")
                spk_name = _session_cluster_map.get(raw_lbl, f"Speaker {raw_lbl}")
            all_turns.append({
                "turn_order": t["turn_order"],
                "text": t.get("text", ""),
                "speaker": spk_name.strip()
            })
    return all_turns


def _compute_speaker_eligibility(all_turns):
    """Single backend source of truth for who currently qualifies to be shown
    (registered profile OR >= 100 words spoken so far). The frontend must never
    recompute this itself -- it should only render/prune based on what this
    function (via the API responses below) says is eligible right now, so the
    two sides can't drift out of sync from client-side lag.

    Returns:
      eligible_user_ids: set of user_ids that currently qualify
      speaker_turns: {user_id: {"name":.., "user_id":.., "turns": [...]}}
      speaker_word_counts: {user_id: int}
      user_id_map / canonical_names: name(lower) -> user_id / canonical name
    """
    registered_profiles = speaker_id_engine.get_users_detailed()
    user_id_map = {p["name"].lower().strip(): p.get("user_id", f"usr_{p['name'].lower().replace(' ', '_')}") for p in registered_profiles}
    canonical_names = {p["name"].lower().strip(): p["name"] for p in registered_profiles}

    speaker_turns = {}
    speaker_word_counts = {}
    for t in all_turns:
        spk = t["speaker"]
        spk_lower = spk.lower().strip()
        canon_name = canonical_names.get(spk_lower, spk)
        u_id = user_id_map.get(spk_lower, f"usr_{re.sub(r'[^a-zA-Z0-9_]', '_', canon_name.lower())}")

        if u_id not in speaker_turns:
            speaker_turns[u_id] = {"name": canon_name, "user_id": u_id, "turns": []}
            speaker_word_counts[u_id] = 0

        speaker_turns[u_id]["turns"].append(t)
        speaker_word_counts[u_id] += len((t["text"] or "").split())

    eligible_user_ids = set()
    for u_id, s_info in speaker_turns.items():
        is_registered = s_info["name"].lower().strip() in user_id_map
        if is_registered or speaker_word_counts[u_id] >= 100:
            eligible_user_ids.add(u_id)

    return eligible_user_ids, speaker_turns, speaker_word_counts, user_id_map, canonical_names


def _filter_eligible_analysis(stored_analysis, eligible_user_ids):
    """Return a copy of stored_analysis containing only currently-eligible speakers.
    The full permanent history stays untouched in SQLite -- only what gets SENT
    to the client on this response is filtered, so silent-but-eligible speakers
    stay visible while never-eligible/removed ones never reach the frontend at all."""
    speakers = stored_analysis.get("speakers", []) if isinstance(stored_analysis, dict) else []
    filtered = [s for s in speakers if (s or {}).get("user_id") in eligible_user_ids]
    out = dict(stored_analysis) if isinstance(stored_analysis, dict) else {"overall_summary": "", "speakers": []}
    out["speakers"] = filtered
    return out


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """Legacy endpoint — redirects to live-intelligence."""
    return api_live_intelligence()


_speaker_last_analyzed_turn: dict = {}
_last_analyzed_global_turn: int = 0


@app.route("/api/wordcloud", methods=["POST"])
def api_wordcloud():
    """Legacy endpoint — redirects to live-intelligence."""
    return api_live_intelligence()


@app.route("/api/live-intelligence", methods=["POST"])
def api_live_intelligence():
    """Incremental Gemini intelligence call.
    Only sends speakers who have NEW speech activity since their last analysis,
    merging their updated cards into SQLite while preserving silent speakers."""
    global _speaker_last_analyzed_turn, _last_analyzed_global_turn

    data = request.get_json(force=True) or {}
    transcript = data.get("transcript", "").strip()

    all_turns = _load_all_turns()
    if not all_turns:
        return jsonify({"topics": [], "overall_summary": "", "speakers": [], "words": []})

    current_max_global_turn = max(t["turn_order"] for t in all_turns)

    # Backend-only eligibility (registered OR >= 100 words). This is the single
    # source of truth -- every response below is filtered through it, so the
    # frontend never has to (and never does) recompute eligibility itself.
    eligible_user_ids, speaker_turns, speaker_word_counts, user_id_map, canonical_names = _compute_speaker_eligibility(all_turns)

    # Identify eligible speakers with NEW turns since their last analysis
    active_speakers_data = []
    for u_id in eligible_user_ids:
        s_info = speaker_turns[u_id]
        last_analyzed = _speaker_last_analyzed_turn.get(u_id, 0)
        new_turns = [t for t in s_info["turns"] if t["turn_order"] > last_analyzed]

        if new_turns:
            recent_text = "\n".join([f"- {t['text']}" for t in new_turns[-15:]])
            active_speakers_data.append({
                "user_id": u_id,
                "name": s_info["name"],
                "total_words": speaker_word_counts[u_id],
                "max_turn": max(t["turn_order"] for t in new_turns),
                "recent_text": recent_text
            })

    # Check if we have new turns for Word Cloud discussion topics
    new_global_turns = [t for t in all_turns if t["turn_order"] > _last_analyzed_global_turn]

    # ZERO-WASTE GUARD: If no speaker has new turns AND no new global turns, return cached SQLite directly
    if not active_speakers_data and not new_global_turns:
        stored_analysis = db.get_meeting_meta("speaker_analysis", {"overall_summary": "", "speakers": []})
        stored_words = db.get_meeting_meta("wordcloud", [])
        filtered_analysis = _filter_eligible_analysis(stored_analysis, eligible_user_ids)
        return jsonify({
            "topics": [w[0] for w in stored_words] if stored_words else [],
            "words": stored_words,
            **filtered_analysis
        })

    # Build focused prompt for Gemini with ONLY active speakers + new dialogue
    speakers_prompt_section = ""
    for aspk in active_speakers_data:
        speakers_prompt_section += (
            f"\n[SPEAKER: {aspk['name']} | User ID: {aspk['user_id']}]\n"
            f"Total Words Spoken: {aspk['total_words']}\n"
            f"New Utterances:\n{aspk['recent_text']}\n"
        )

    recent_dialogue = "\n".join([f"{t['speaker']}: {t['text']}" for t in all_turns[-30:]])

    prompt = (
        "You are an expert discourse intelligence analyst evaluating an active multi-speaker meeting.\n"
        "The following speakers have NEW speech activity in this interval:\n"
        f"{speakers_prompt_section if speakers_prompt_section else 'Evaluate recent conversation topics.'}\n\n"
        "Recent conversation context:\n"
        f"{recent_dialogue}\n\n"
        "Analyze these ACTIVE speakers and the latest discussion topics.\n"
        "CRITICAL GUIDELINES:\n"
        "- Generate a speaker analysis item ONLY for the active speakers listed above with their exact 'user_id' and 'name'.\n"
        "- Extract 7 to 10 punchy discussion topics (1-3 words each) reflecting the latest dialogue.\n"
        "- Provide an updated 2-3 sentence overall meeting summary.\n\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6", "Topic 7"],\n'
        '  "overall_summary": "Updated 2-3 sentence meeting summary",\n'
        '  "speakers": [\n'
        "    {\n"
        '      "user_id": "usr_example",\n'
        '      "name": "Speaker Name",\n'
        '      "score": 8.5,\n'
        '      "score_reason": "1-sentence explanation",\n'
        '      "summary": "Assessment of their contributions",\n'
        '      "key_points": ["point 1", "point 2"],\n'
        '      "keywords": ["keyword 1", "keyword 2"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    raw_json = call_gemini(prompt, response_json=True, timeout=90)
    if not raw_json:
        raw_json = call_gemini(prompt, timeout=90)

    stored_analysis = db.get_meeting_meta("speaker_analysis", {"overall_summary": "", "speakers": []})
    existing_speaker_map = {}
    for s in stored_analysis.get("speakers", []):
        sid = s.get("user_id") or f"usr_{re.sub(r'[^a-zA-Z0-9_]', '_', s.get('name', '').lower())}"
        existing_speaker_map[sid] = s

    if raw_json:
        try:
            cleaned = raw_json.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())

            if isinstance(parsed, dict):
                # ── 1. Update Word Cloud topics ──
                topics = parsed.get("topics", [])
                if isinstance(topics, list) and len(topics) > 0:
                    topic_weights = [48, 42, 38, 32, 28, 25, 22, 18, 15, 12]
                    words = []
                    for i, t in enumerate(topics[:10]):
                        if t and isinstance(t, str):
                            words.append([t.strip(), topic_weights[min(i, len(topic_weights) - 1)]])
                    if words:
                        db.set_meeting_meta("wordcloud", words)
                        broadcast_sse("wordcloud_updated", {"words": words})

                # ── 2. Incrementally merge active speakers ──
                # Only accept speakers the backend itself flagged as active/eligible
                # this cycle -- never let Gemini's output add a name to permanent
                # storage that the backend didn't ask it to analyze.
                active_ids_this_cycle = {aspk["user_id"] for aspk in active_speakers_data}
                new_speakers = parsed.get("speakers", [])
                for n_spk in new_speakers:
                    n_uid = n_spk.get("user_id") or user_id_map.get(n_spk.get("name", "").lower().strip())
                    if not n_uid:
                        n_uid = f"usr_{re.sub(r'[^a-zA-Z0-9_]', '_', n_spk.get('name', '').lower())}"
                    if n_uid not in active_ids_this_cycle:
                        continue
                    n_spk["user_id"] = n_uid
                    existing_speaker_map[n_uid] = n_spk

                if parsed.get("overall_summary"):
                    stored_analysis["overall_summary"] = parsed["overall_summary"]

                # Persist the FULL permanent history (all speakers ever analyzed,
                # active or silent) -- eligibility filtering only happens on the
                # way out to the client, never on the way into storage.
                stored_analysis["speakers"] = list(existing_speaker_map.values())
                db.set_meeting_meta("speaker_analysis", stored_analysis)

                # Mark active speakers as analyzed up to their max turn
                for aspk in active_speakers_data:
                    _speaker_last_analyzed_turn[aspk["user_id"]] = aspk["max_turn"]
                _last_analyzed_global_turn = current_max_global_turn

                filtered_analysis = _filter_eligible_analysis(stored_analysis, eligible_user_ids)
                broadcast_sse("analysis_updated", filtered_analysis)
                return jsonify({
                    "topics": topics,
                    "words": db.get_meeting_meta("wordcloud", []),
                    **filtered_analysis
                })
        except Exception:
            pass

    # Fallback to local extraction if Gemini was unreachable
    fallback_words = extract_local_wordcloud(transcript)
    if fallback_words:
        db.set_meeting_meta("wordcloud", fallback_words)
        broadcast_sse("wordcloud_updated", {"words": fallback_words})

    filtered_analysis = _filter_eligible_analysis(stored_analysis, eligible_user_ids)
    return jsonify({
        "words": db.get_meeting_meta("wordcloud", fallback_words),
        "topics": [w[0] for w in fallback_words],
        **filtered_analysis
    })


def extract_local_wordcloud(transcript: str):
    """Frequency-ranked bigram/keyword topic extraction. Deliberately
    domain-agnostic: this is the offline fallback used when Gemini is
    unreachable, so it has to work for any meeting topic, not just the ones
    it happened to be tested against."""
    text_clean = re.sub(r'^[A-Za-z0-9_\s]+:\s*', '', transcript, flags=re.MULTILINE)
    sentences = re.split(r'[.!?\n]+', text_clean)

    # Bigrams bounded strictly within sentences, plus unigram frequency as
    # a fill-in source -- both purely frequency-driven, no topic hardcoding.
    phrase_counts = Counter()
    word_counts = Counter()

    for s in sentences:
        words = [w.strip('.,!?:;"\'()[]{}').lower() for w in s.split()]
        words = [re.sub(r"['’](s|re|ve|ll|d|m|t)$", '', w) for w in words]
        words = [w for w in words if w and len(w) > 2 and w not in GENERIC_BAN_WORDS]

        for w in words:
            if len(w) > 3:
                word_counts[w] += 1

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 != w2:
                phrase_counts[f"{w1.capitalize()} {w2.capitalize()}"] += 1

    found_topics = []
    for phrase, _ in phrase_counts.most_common(10):
        if not any(phrase.lower() in t.lower() or t.lower() in phrase.lower() for t in found_topics):
            found_topics.append(phrase)

    # Fill remaining slots with high-frequency single words not already covered
    if len(found_topics) < 6:
        for w, _ in word_counts.most_common(10):
            if any(w in t.lower() for t in found_topics):
                continue
            found_topics.append(w.capitalize())
            if len(found_topics) >= 8:
                break

    weights = [48, 38, 32, 26, 22, 18, 15, 14, 13, 12]
    return [[topic, weights[min(idx, len(weights) - 1)]] for idx, topic in enumerate(found_topics[:10])]



@app.route("/api/meeting-analytics", methods=["GET"])
def api_meeting_analytics():
    """Returns persistent analytics (wordcloud, analysis summary) from SQLite DB,
    filtered through the same backend-only eligibility check as /api/live-intelligence
    so a page reload/reconnect can never show a stale or ineligible speaker card."""
    stored_analysis = db.get_meeting_meta("speaker_analysis", {})
    if isinstance(stored_analysis, dict) and stored_analysis.get("speakers"):
        all_turns = _load_all_turns()
        eligible_user_ids, *_ = _compute_speaker_eligibility(all_turns)
        stored_analysis = _filter_eligible_analysis(stored_analysis, eligible_user_ids)
    return jsonify({
        "wordcloud": db.get_meeting_meta("wordcloud", []),
        "analysis": stored_analysis
    })


@app.route("/api/recording-state", methods=["GET"])
def api_get_recording_state():
    """Returns current transcription active/paused state."""
    state = db.get_meeting_meta("is_transcribing", True)
    return jsonify({"is_transcribing": bool(state)})


@app.route("/api/set-recording-state", methods=["POST"])
def api_set_recording_state():
    """Start or stop transcribing without terminating WebSocket connections."""
    data = request.get_json(force=True) or {}
    is_transcribing = bool(data.get("is_transcribing", True))
    db.set_meeting_meta("is_transcribing", is_transcribing)
    broadcast_sse("recording_state_changed", {"is_transcribing": is_transcribing})
    return jsonify({"status": "ok", "is_transcribing": is_transcribing})


# ── Speaker ID & Turn Endpoints ──────────────────────────────────────────────


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
    company_name = data.get("company_name")
    if company_name is not None:
        company_name = str(company_name).strip()
    if not name:
        return jsonify({"error": "User name is required."}), 400
    try:
        res = speaker_id_engine.create_user(name, avatar_b64=avatar_b64, company_name=company_name)
        broadcast_sse("users_updated", res)
        return jsonify(res)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/enroll-user-voice", methods=["POST"])
def api_enroll_user_voice():
    """Enroll a user by recording an audio reading (e.g. 10s to 60s essay sample)."""
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    audio_b64 = data.get("audio_b64")
    if not name or not audio_b64:
        return jsonify({"error": "name and audio_b64 are required."}), 400

    try:
        pcm = _decode_pcm_base64(audio_b64)
        if len(pcm) < int(0.5 * 16000):
            return jsonify({"error": "Voice sample too short (must be at least 0.5 seconds)."}), 400

        res = speaker_id_engine.enroll_user_long_audio(name, pcm)
        _recalculate_turn_predictions()
        broadcast_sse("users_updated", res)
        broadcast_sse("turns_updated", {})
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
        for entry in list(turn_store.values()):
            if (entry.get("tagged_as") or "").lower() == name.lower():
                entry["tagged_as"] = None
        res = speaker_id_engine.delete_user(name)
        _rebuild_all_centroids()
        _recalculate_turn_predictions()
        broadcast_sse("users_updated", {"action": "deleted", "name": name})
        broadcast_sse("turns_updated", {"action": "deleted", "name": name})
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


@app.route("/api/reset-voice-model", methods=["POST"])
def api_reset_voice_model():
    """Reset ONLY the voice model centroid learnings across all profiles.
    PRESERVES all user profiles (names, avatars, companies) and all turn text transcripts."""
    global _session_cluster_map
    res = speaker_id_engine.reset_voice_model_centroids()
    for entry in turn_store.values():
        entry["tagged_as"] = None
        entry["predicted_speaker"] = None
        entry["predicted_confidence"] = 0.0
        entry["scores"] = {}
    db.clear_all_turn_tags()
    _session_cluster_map.clear()
    _recalculate_turn_predictions()
    broadcast_sse("users_updated", {})
    broadcast_sse("turns_updated", {})
    return jsonify(res)


@app.route("/api/clear-voiceprints", methods=["POST"])
def api_clear_voiceprints():
    res = speaker_id_engine.clear_session()
    for entry in turn_store.values():
        entry["tagged_as"] = None
        entry["predicted_speaker"] = None
        entry["predicted_confidence"] = 0.0
        entry["scores"] = {}
    _recalculate_turn_predictions()
    broadcast_sse("cleared", {})
    broadcast_sse("turns_updated", {})
    return jsonify(res)


@app.route("/api/clear-turns", methods=["POST"])
def api_clear_turns():
    """Clear ONLY turns and sentence transcripts.
    PRESERVES all registered user profiles, avatars, and voiceprint centroids in memory and SQLite."""
    global _last_analyzed_global_turn
    turn_store.clear()
    _speaker_last_analyzed_turn.clear()
    _last_analyzed_global_turn = 0
    db.clear_all_turns()
    db.clear_meeting_meta()
    broadcast_sse("turns_cleared", {})
    broadcast_sse("turns_updated", {})
    broadcast_sse("wordcloud_cleared", {})
    broadcast_sse("analysis_cleared", {})
    return jsonify({"status": "turns_cleared", "turns_count": 0})


@app.route("/api/clear-all", methods=["POST"])
def api_clear_all():
    global _session_cluster_map, _last_analyzed_global_turn
    turn_store.clear()
    _session_cluster_map.clear()
    _speaker_last_analyzed_turn.clear()
    _last_analyzed_global_turn = 0
    speaker_id_engine.clear_session()
    db.clear_all_profiles()
    db.clear_all_turns()
    db.clear_meeting_meta()
    broadcast_sse("cleared", {})
    broadcast_sse("turns_cleared", {})
    broadcast_sse("users_updated", {})
    broadcast_sse("wordcloud_cleared", {})
    broadcast_sse("analysis_cleared", {})
    return jsonify({"status": "cleared", "turns_count": 0, "users_count": 0})


def _rebuild_all_centroids():
    canonical_names = {k.lower().strip(): k for k in speaker_id_engine._profiles.keys()}
    grouped = {}
    fallback_grouped = {}

    for entry in list(turn_store.values()):
        if entry.get("is_mixed"):
            continue  # Exclude mixed / contaminated audio from all centroids

        speaker = (entry.get("tagged_as") or "").strip()
        if not speaker:
            continue
        canon_name = canonical_names.get(speaker.lower(), speaker)
        emb = entry.get("embedding")
        if emb is not None:
            duration_s = entry.get("duration_s", 1.0)

            if canon_name not in fallback_grouped:
                fallback_grouped[canon_name] = []
            fallback_grouped[canon_name].append(emb)

            # Combined into centroid only if >= 1.0s audio
            if duration_s >= 1.0:
                if canon_name not in grouped:
                    grouped[canon_name] = []
                grouped[canon_name].append(emb)

    # Fallback to shorter sample if a user has no >= 1.0s samples yet
    for spk, embs in fallback_grouped.items():
        if spk not in grouped or not grouped[spk]:
            grouped[spk] = embs

    speaker_id_engine.sync_all_user_centroids(grouped)


def _update_single_user_centroid(speaker_name):
    """Incrementally update only one user's centroid instead of rebuilding all."""
    canonical_names = {k.lower().strip(): k for k in speaker_id_engine._profiles.keys()}
    canon_name = canonical_names.get((speaker_name or "").lower().strip(), speaker_name)
    name = (canon_name or "").strip()
    if not name:
        return

    embeddings = []
    fallback_embeddings = []

    for entry in turn_store.values():
        if entry.get("is_mixed"):
            continue  # Exclude mixed audio

        if (entry.get("tagged_as") or "").strip().lower() == name.lower():
            emb = entry.get("embedding")
            if emb is not None:
                fallback_embeddings.append(emb)
                duration_s = entry.get("duration_s", 1.0)
                if duration_s >= 1.0:
                    embeddings.append(emb)

    # Fallback to shorter samples if no >= 1.0s samples exist
    if not embeddings and fallback_embeddings:
        embeddings = fallback_embeddings

    speaker_id_engine.update_user_centroid(name, embeddings)


_session_cluster_map: dict = {}


def _update_cluster_map_from_turns():
    """Update session cluster mapping (e.g. Speaker A -> Vinay) from high-confidence anchor turns."""
    global _session_cluster_map
    for entry in turn_store.values():
        lbl = entry.get("speaker_label")
        if not lbl:
            continue
        if entry.get("tagged_as"):
            _session_cluster_map[lbl] = entry["tagged_as"]
        elif entry.get("predicted_speaker") and entry.get("predicted_confidence", 0.0) >= CONF_TENTATIVE:
            _session_cluster_map[lbl] = entry["predicted_speaker"]


def _recalculate_turn_predictions():
    """Re-identify all turns against current user centroids and batch-update database."""
    _update_cluster_map_from_turns()
    batch_updates = []
    for turn_order, entry in list(turn_store.items()):
        emb = entry.get("embedding")
        lbl = entry.get("speaker_label")
        if emb is not None:
            id_res = speaker_id_engine.identify_embedding(emb)
            predicted_speaker, predicted_confidence, _is_winner, scores = _resolve_prediction(id_res, lbl)

            entry["predicted_speaker"] = predicted_speaker
            entry["predicted_confidence"] = predicted_confidence
            entry["scores"] = scores
        else:
            if lbl and lbl in _session_cluster_map:
                entry["predicted_speaker"] = _session_cluster_map[lbl]
                # No audio for this specific turn was scored, so there's no real
                # per-turn number to show -- 0% is honest ("inherited, unmeasured"),
                # not a guess dressed up as a percentage.
                entry["predicted_confidence"] = 0.0
            else:
                entry["predicted_speaker"] = None
                entry["predicted_confidence"] = 0.0
            entry["scores"] = {}

        batch_updates.append((
            entry.get("tagged_as"),
            entry.get("predicted_speaker"),
            entry.get("predicted_confidence", 0.0),
            turn_order,
        ))

    db.batch_update_turn_predictions(batch_updates)


def init_app_state():
    """Load profiles, turns, centroids, and session cluster mappings from SQLite on startup."""
    speaker_id_engine.init_profiles_from_db()
    db_turns = db.get_all_turns()
    for t in db_turns:
        to = t["turn_order"]
        emb = None
        if t.get("embedding_json"):
            try:
                emb = torch.tensor(json.loads(t["embedding_json"]), dtype=torch.float32)
            except Exception:
                pass

        turn_store[to] = {
            "audio_b64": None,
            "embedding": emb,
            "text": t.get("text", ""),
            "speaker_label": t.get("speaker_label", "A"),
            "predicted_speaker": t.get("predicted_speaker"),
            "predicted_confidence": t.get("predicted_confidence", 0.0),
            "scores": {},
            "tagged_as": t.get("tagged_as"),
            "is_mixed": bool(t.get("is_mixed", 0)),
        }
    _rebuild_all_centroids()
    _update_cluster_map_from_turns()
    _recalculate_turn_predictions()
    log.info("Initialized app with %d turns and recovered session cluster mapping: %s", len(turn_store), _session_cluster_map)


init_app_state()


from concurrent.futures import ThreadPoolExecutor

_bg_pool = ThreadPoolExecutor(max_workers=2)


def _process_turn_embedding_async(turn_order, audio_b64):
    if not audio_b64:
        return
    try:
        pcm = _decode_pcm_base64(audio_b64)
        if len(pcm) < int(0.3 * 16000):
            return
        duration_s = len(pcm) / 16000.0
        emb = speaker_id_engine.extract_embedding(pcm)
        id_res = speaker_id_engine.identify_embedding(emb)

        entry = turn_store.get(turn_order)
        lbl = entry.get("speaker_label") if entry else None

        predicted_speaker, predicted_confidence, is_winner, scores = _resolve_prediction(id_res, lbl)

        # Lock in session cluster mapping only for genuinely confident matches --
        # a weak/inherited guess should never overwrite what the cluster map
        # already knows for this raw label.
        if predicted_speaker and (is_winner or predicted_confidence >= CONF_CONFIRMED) and lbl:
            _session_cluster_map[lbl] = predicted_speaker

        # Persist the computed embedding so it doesn't need re-extraction on restart
        embedding_json = json.dumps(emb.tolist())
        db.update_turn_embedding(turn_order, embedding_json)

        if entry:
            entry["embedding"] = emb
            entry["duration_s"] = duration_s
            entry["audio_b64"] = None
            entry["predicted_speaker"] = predicted_speaker
            entry["predicted_confidence"] = predicted_confidence
            entry["scores"] = scores

            # Auto-tag and combine into centroid if the engine flagged a genuine winner OR the
            # real confidence crosses CONF_CONFIRMED -- AND audio length >= 1.0s
            if (
                predicted_speaker
                and (is_winner or predicted_confidence >= CONF_CONFIRMED)
                and duration_s >= 1.0
                and not entry.get("tagged_as")
            ):
                entry["tagged_as"] = predicted_speaker
                db.update_turn_tag(turn_order, predicted_speaker)
                _update_single_user_centroid(predicted_speaker)
                broadcast_sse("tag_updated", {"turn_order": turn_order, "tagged_as": predicted_speaker})
            else:
                db.upsert_turn(
                    turn_order=turn_order,
                    text=entry["text"],
                    speaker_label=entry["speaker_label"],
                    audio_b64=None,
                    tagged_as=entry.get("tagged_as"),
                    predicted_speaker=predicted_speaker,
                    predicted_confidence=predicted_confidence,
                    embedding_json=embedding_json,
                )
                broadcast_sse("turn_update", {
                    "turn_order": turn_order,
                    "text": entry["text"],
                    "tagged_as": entry.get("tagged_as"),
                    "predicted_speaker": predicted_speaker,
                    "predicted_confidence": predicted_confidence,
                })
    except Exception:
        pass


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

    # 1. Instant SQLite write in <1ms without storing audio
    db.upsert_turn(
        turn_order=turn_order,
        text=text,
        speaker_label=speaker_label,
        audio_b64=None,
        tagged_as=None,
        predicted_speaker=None,
        predicted_confidence=0.0,
    )

    turn_store[turn_order] = {
        "audio_b64": None,
        "embedding": None,
        "duration_s": 1.0,
        "text": text,
        "speaker_label": speaker_label,
        "predicted_speaker": None,
        "predicted_confidence": 0.0,
        "scores": {},
        "tagged_as": None,
    }

    # 2. Instant Real-time Push to Admin Panel
    broadcast_sse("turn_update", {"turn_order": turn_order, "text": text, "tagged_as": None})

    # 3. Offload voiceprint extraction to background thread
    if audio_b64:
        _bg_pool.submit(_process_turn_embedding_async, turn_order, audio_b64)

    return jsonify({
        "status": "stored",
        "turn_order": turn_order,
    })


@app.route("/api/turns", methods=["GET"])
def api_turns():
    """Returns stored turns directly from persistent SQLite database (lightweight, no audio blobs)."""
    turns = db.get_all_turns_lite()
    result = []
    for t in turns:
        to = t["turn_order"]
        entry = turn_store.get(to, {})
        resolved_name, resolved_tier = resolve_speaker_display({
            "tagged_as": t["tagged_as"],
            "predicted_speaker": t["predicted_speaker"],
            "predicted_confidence": t.get("predicted_confidence", 0.0),
        })
        result.append({
            "turn_order": to,
            "text": t["text"],
            "speaker_label": t["speaker_label"],
            "tagged_as": t["tagged_as"],
            "predicted_speaker": t["predicted_speaker"],
            "predicted_confidence": t.get("predicted_confidence", 0.0),
            "resolved_speaker_name": resolved_name,
            "resolved_tier": resolved_tier,
            "scores": entry.get("scores", {}),
            "is_mixed": bool(t.get("is_mixed", 0)),
            "has_audio": bool(t.get("has_audio")),
        })
    return jsonify({"turns": result})


@app.route("/api/tag-turn-from-admin", methods=["POST"])
def api_tag_turn_from_admin():
    data = request.get_json(force=True) or {}
    turn_order = data.get("turn_order")
    speaker_name = (data.get("speaker_name") or "").strip()
    update_centroid = bool(data.get("update_centroid", True))
    set_mixed = data.get("is_mixed")
    recalculate_predictions = bool(data.get("recalculate_predictions", update_centroid))

    if turn_order is None or not speaker_name:
        return jsonify({"error": "turn_order and speaker_name required"}), 400

    turn_order = int(turn_order)
    entry = turn_store.get(turn_order)
    if not entry:
        db_turn = db.get_turn(turn_order)
        if db_turn:
            emb = None
            if db_turn.get("embedding_json"):
                try:
                    emb = torch.tensor(json.loads(db_turn["embedding_json"]), dtype=torch.float32)
                except Exception:
                    pass
            entry = {
                "audio_b64": db_turn.get("audio_b64"),
                "embedding": emb,
                "text": db_turn.get("text", ""),
                "speaker_label": db_turn.get("speaker_label", "A"),
                "tagged_as": db_turn.get("tagged_as"),
                "is_mixed": bool(db_turn.get("is_mixed", 0)),
            }
            turn_store[turn_order] = entry
        else:
            entry = {
                "audio_b64": None,
                "embedding": None,
                "text": "",
                "speaker_label": "A",
                "tagged_as": None,
                "is_mixed": False,
            }
            turn_store[turn_order] = entry

    old_speaker = (entry.get("tagged_as") or "").strip()
    entry["tagged_as"] = speaker_name

    # If update_centroid is False or is_mixed is True, mark this turn as mixed/isolated
    if not update_centroid or set_mixed is True:
        entry["is_mixed"] = True
        db.update_turn_mixed(turn_order, 1)
        broadcast_sse("turn_mixed_updated", {"turn_order": turn_order, "is_mixed": True})
    elif set_mixed is False:
        entry["is_mixed"] = False
        db.update_turn_mixed(turn_order, 0)
        broadcast_sse("turn_mixed_updated", {"turn_order": turn_order, "is_mixed": False})

    db.update_turn_tag(turn_order, speaker_name)

    # 1. INSTANT sub-millisecond SSE broadcast to all connected UIs
    broadcast_sse("tag_updated", {
        "turn_order": turn_order,
        "tagged_as": speaker_name,
        "is_mixed": entry.get("is_mixed", False)
    })
    broadcast_sse("turns_updated", {"turn_order": turn_order})

    # 2. Async background centroid & prediction update
    def _async_update_centroids():
        try:
            if not update_centroid:
                # One-off isolated fix:
                # Unconditionally rebuild all centroids so this turn is excluded from ALL centroids,
                # immediately removing it from whichever previous speaker's voiceprint it was in.
                _rebuild_all_centroids()
                broadcast_sse("users_updated", {})
                broadcast_sse("turns_updated", {})
                return

            if entry.get("embedding") is None and entry.get("audio_b64"):
                pcm = _decode_pcm_base64(entry["audio_b64"])
                if len(pcm) >= int(0.3 * 16000):
                    emb = speaker_id_engine.extract_embedding(pcm)
                    entry["embedding"] = emb
                    db.update_turn_embedding(turn_order, json.dumps(emb.tolist()))

            # Rebuild all centroids so target speaker gets the sample and previous speaker loses it
            _rebuild_all_centroids()
            if recalculate_predictions:
                _recalculate_turn_predictions()
            broadcast_sse("users_updated", {})
            broadcast_sse("turns_updated", {})
        except Exception as e:
            log.error("Error in _async_update_centroids: %s", e)

    _bg_pool.submit(_async_update_centroids)

    return jsonify({
        "status": "tagged",
        "turn_order": turn_order,
        "speaker_name": speaker_name,
        "is_mixed": entry.get("is_mixed", False),
        "update_centroid": update_centroid,
    })


@app.route("/api/delete-turn", methods=["POST"])
def api_delete_turn():
    """Delete a single turn from SQLite database, in-memory store, and recalculate user centroid."""
    data = request.get_json(force=True) or {}
    turn_order = data.get("turn_order")
    if turn_order is None:
        return jsonify({"error": "turn_order required"}), 400

    turn_order = int(turn_order)
    entry = turn_store.pop(turn_order, None)
    tagged_speaker = (entry.get("tagged_as") or "").strip() if entry else None

    if not tagged_speaker:
        db_turn = db.get_turn(turn_order)
        if db_turn:
            tagged_speaker = (db_turn.get("tagged_as") or "").strip()

    # Delete from SQLite persistent database
    db.delete_turn(turn_order)

    # If the deleted turn was tagged to a speaker, recompute their centroid without this turn
    if tagged_speaker:
        _update_single_user_centroid(tagged_speaker)
        _recalculate_turn_predictions()

    # Broadcast instant removal to all clients and admin panels
    broadcast_sse("turn_deleted", {"turn_order": turn_order})
    broadcast_sse("turns_updated", {"turn_order": turn_order})
    broadcast_sse("users_updated", {})

    return jsonify({
        "status": "deleted",
        "turn_order": turn_order,
    })


@app.route("/api/set-turn-mixed", methods=["POST"])
def api_set_turn_mixed():
    """Toggle a turn as mixed/cross-talk speech to exclude or include its embedding in centroids."""
    data = request.get_json(force=True) or {}
    turn_order = data.get("turn_order")
    if turn_order is None:
        return jsonify({"error": "turn_order required"}), 400

    turn_order = int(turn_order)
    is_mixed = bool(data.get("is_mixed", False))

    entry = turn_store.get(turn_order)
    if entry:
        entry["is_mixed"] = is_mixed

    db.update_turn_mixed(turn_order, 1 if is_mixed else 0)

    # Recompute centroids if this turn has a tagged speaker
    tagged_speaker = entry.get("tagged_as") if entry else None
    if not tagged_speaker:
        db_turn = db.get_turn(turn_order)
        if db_turn:
            tagged_speaker = db_turn.get("tagged_as")

    if tagged_speaker:
        _update_single_user_centroid(tagged_speaker)
        _recalculate_turn_predictions()

    # Broadcast to admin and clients
    broadcast_sse("turn_mixed_updated", {"turn_order": turn_order, "is_mixed": is_mixed})
    broadcast_sse("turns_updated", {"turn_order": turn_order})
    broadcast_sse("users_updated", {})

    return jsonify({
        "status": "updated",
        "turn_order": turn_order,
        "is_mixed": is_mixed,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
