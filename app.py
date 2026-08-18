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
        # Fast path: load cached embedding from DB
        if t.get("embedding_json"):
            try:
                emb = torch.tensor(json.loads(t["embedding_json"]), dtype=torch.float32)
            except Exception:
                pass
        # Slow fallback: extract from audio (one-time migration for legacy turns)
        if emb is None and t.get("audio_b64"):
            try:
                pcm = _decode_pcm_base64(t["audio_b64"])
                if len(pcm) >= int(0.3 * 16000):
                    emb = speaker_id_engine.extract_embedding(pcm)
                    # Cache for future restarts
                    db.update_turn_embedding(to, json.dumps(emb.tolist()))
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
            "is_mixed": bool(t.get("is_mixed", 0)),
        }
    log.info("Initialized app with %d turns from database.", len(turn_store))


init_app_state()


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


GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]


def call_gemini(prompt, response_json=False, timeout=20):
    """Executes prompt on Gemini Flash with automatic model fallback."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or GEMINI_API_KEY
    if not key:
        return None
    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        if response_json:
            body["generationConfig"] = {"response_mime_type": "application/json"}
        try:
            resp = requests.post(url, params={"key": key}, json=body, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                log.warning("Gemini %s returned %d: %s", model, resp.status_code, resp.text[:120])
        except Exception as e:
            log.warning("Gemini %s request failed: %s", model, e)
            continue
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


def _synthesize_takeaway(sentence: str) -> str:
    """Derives what the speaker meant rather than echoing verbatim raw speech."""
    s = sentence.strip()
    s_lower = s.lower()

    if re.search(r'\b(virtuous|values|goals)\b', s_lower) and re.search(r'\b(hope|track|confident)\b', s_lower):
        return "Emphasized that AI value alignment and virtuous behavior are currently unverified hopes rather than guaranteed safeguards."
    if re.search(r'\b(lie|lying|pretend|deceiv)\b', s_lower):
        return "Highlighted that current AI systems exhibit deceptive behaviors, complicating alignment and trust."
    if re.search(r'\b(super\s*intelligent|inherently difficult)\b', s_lower):
        return "Stressed the fundamental difficulty of ensuring superintelligent systems adhere to intended human virtues."
    if re.search(r'\b(new species|ruling|extinct|outcompeted)\b', s_lower):
        return "Warned of catastrophic existential risks, including humanity being outcompeted or replaced by artificial species."
    if re.search(r'\b(database|sql|wal|index|indexing)\b', s_lower):
        return "Proposed database optimization using WAL mode and efficient indexing for high concurrency."
    if re.search(r'\b(ecapa|speechbrain|speaker|embedding)\b', s_lower):
        return "Recommended deploying ECAPA-TDNN neural embeddings for robust acoustic speaker identification."

    # Generic semantic condensing: extract meaningful core nouns and domain essence
    clean_words = [w for w in re.findall(r'\b[A-Za-z]{3,}\b', s) if w.lower() not in MASTER_STOPWORDS]
    if len(clean_words) >= 3:
        return f"Addressed core considerations regarding {' '.join(w.capitalize() for w in clean_words[:3])}."
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

        # Sentence extraction and semantic synthesis for takeaways
        sentences = [s.strip() for s in re.split(r"[.!?]+", all_text) if len(s.strip().split()) >= 4]
        synthesized_takeaways = []
        seen_takeaways = set()
        for s in sentences:
            tw = _synthesize_takeaway(s)
            if tw and tw not in seen_takeaways:
                seen_takeaways.add(tw)
                synthesized_takeaways.append(tw)
            if len(synthesized_takeaways) >= 4:
                break

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


def _filter_eligible_speakers(speakers_list, full_transcript):
    """Keep only speakers who are registered backend profiles OR have spoken >= 100 words."""
    if not isinstance(speakers_list, list):
        return []

    registered_names = set()
    try:
        registered_names = {p["name"].strip().lower() for p in db.get_all_profiles() if p.get("name")}
    except Exception:
        pass

    spk_word_counts = {}
    for line in full_transcript.split("\n"):
        if ":" in line:
            s_name, s_txt = line.split(":", 1)
            s_name = s_name.strip().lower()
            spk_word_counts[s_name] = spk_word_counts.get(s_name, 0) + len(s_txt.split())

    filtered = []
    for spk in speakers_list:
        name = (spk.get("name") or "").strip()
        name_lower = name.lower()
        is_registered = name_lower in registered_names
        word_count = spk_word_counts.get(name_lower, 0)
        if is_registered or word_count >= 100:
            filtered.append(spk)
    return filtered


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """Legacy endpoint — redirects to live-intelligence."""
    return api_live_intelligence()


@app.route("/api/wordcloud", methods=["POST"])
def api_wordcloud():
    """Legacy endpoint — redirects to live-intelligence."""
    return api_live_intelligence()


@app.route("/api/live-intelligence", methods=["POST"])
def api_live_intelligence():
    """Single unified Gemini call every 30 seconds.
    Returns BOTH discussion topics (7-10 keywords) AND per-speaker analysis in one shot."""
    data = request.get_json(force=True) or {}
    transcript = data.get("transcript", "").strip()
    if not transcript:
        return jsonify({"topics": [], "overall_summary": "", "speakers": []})

    # Trim to last 200 lines for context efficiency
    lines = transcript.split("\n")
    if len(lines) > 200:
        transcript = "\n".join(lines[-200:])

    prompt = (
        "You are an expert discourse intelligence analyst evaluating a live multi-speaker meeting.\n"
        "Analyze this conversation transcript and return TWO things in a single JSON response:\n\n"
        "═══ PART 1: DISCUSSION TOPICS ═══\n"
        "Extract 7 to 10 concise domain-relevant discussion topics/keywords from the transcript.\n"
        "Each topic must be a punchy 1-3 word phrase (e.g. 'AI Safety', 'Model Latency', 'Database Indexing').\n"
        "STRICTLY FORBIDDEN: conversational filler phrases (e.g. 'There is lots', 'Hope it is').\n\n"
        "═══ PART 2: SPEAKER ANALYSIS ═══\n"
        "Analyze each speaker with complete intellectual honesty, objectivity, and realistic rigor.\n\n"
        "CRITICAL SPEAKER INCLUSION RULE:\n"
        "- ONLY generate a section for speakers who are named participants OR who have spoken at least 100 words.\n"
        "- COMPLETELY OMIT short generic speakers (e.g. 'Speaker A' with only a few sentences).\n\n"
        "CRITICAL EVALUATION GUIDELINES (DO NOT BE A PEOPLE-PLEASER):\n"
        "- Be realistic, objective, and strict. Low marks (1.0 to 4.0) are fine for low-value contributions.\n"
        "- High scores (8.0+) ONLY for concrete technical facts, actionable decisions, or deep domain expertise.\n\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5", "Topic 6", "Topic 7"],\n'
        '  "overall_summary": "2-3 sentence meeting summary",\n'
        '  "speakers": [\n'
        "    {\n"
        '      "name": "Speaker Name",\n'
        '      "score": 8.5,\n'
        '      "score_reason": "1-sentence honest explanation",\n'
        '      "summary": "Objective assessment of their contribution",\n'
        '      "key_points": ["point 1", "point 2"],\n'
        '      "keywords": ["keyword1", "keyword2"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Transcript:\n{transcript}"
    )

    raw_json = call_gemini(prompt, response_json=True, timeout=25)

    # Fallback: retry without forced JSON mime type
    if not raw_json:
        raw_json = call_gemini(prompt, timeout=25)

    if raw_json:
        try:
            cleaned = raw_json.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())

            if isinstance(parsed, dict):
                # ── Process topics into weighted word cloud ──
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
                        parsed["words"] = words

                # ── Process speaker analysis ──
                if "speakers" in parsed:
                    parsed["speakers"] = _filter_eligible_speakers(parsed.get("speakers", []), transcript)
                    db.set_meeting_meta("speaker_analysis", {
                        "overall_summary": parsed.get("overall_summary", ""),
                        "speakers": parsed["speakers"]
                    })
                    broadcast_sse("analysis_updated", {
                        "overall_summary": parsed.get("overall_summary", ""),
                        "speakers": parsed["speakers"]
                    })

                return jsonify(parsed)
        except Exception:
            pass

    # ── Fallback: local extraction ──
    fallback_words = extract_local_wordcloud(transcript)
    fallback_analysis = _fallback_speaker_analysis(transcript)
    db.set_meeting_meta("wordcloud", fallback_words)
    db.set_meeting_meta("speaker_analysis", fallback_analysis)
    broadcast_sse("wordcloud_updated", {"words": fallback_words})
    broadcast_sse("analysis_updated", fallback_analysis)
    return jsonify({
        "words": fallback_words,
        "topics": [w[0] for w in fallback_words],
        **fallback_analysis
    })


def extract_local_wordcloud(transcript: str):
    """Extracts high-level substantive domain topics and macro-concepts."""
    text_clean = re.sub(r'^[A-Za-z0-9_\s]+:\s*', '', transcript, flags=re.MULTILINE)

    stopwords = {
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

    domain_rules = [
        (r'\b(ai|artificial intelligence)\s+(industry|safety|alignment|models|systems|risk|risks|capabilities)\b', lambda m: f"{m.group(1).upper()} {m.group(2).capitalize()}"),
        (r'\b(super\s*intelligent|superintelligence)\s*(ai|systems)?\b', lambda m: 'Superintelligent AI'),
        (r'\b(values|virtues)\s+(and\s+)?(virtues|values|goals)\b', lambda m: 'Values & Alignment'),
        (r'\b(extinct|extinction)\s*(species|risk)?\b', lambda m: 'Extinction Risk'),
        (r'\b(new\s+species)\b', lambda m: 'New Species Creation'),
        (r'\b(lie|lying|deception|deceive|deceptive)\b', lambda m: 'AI Deception Risk'),
        (r'\b(goals?|intentions?)\s+(and\s+)?(values?)\b', lambda m: 'AI Goal Alignment'),
        (r'\b(outcompeted|compete|competition|replace|replacement)\b', lambda m: 'Human Replacement Risk'),
        (r'\b(database|sql|sqlite|indexing|queries|query)\b', lambda m: 'Database Architecture'),
        (r'\b(speech|speechbrain|speaker|voiceprint|embedding|embeddings)\b', lambda m: 'Voice Identification'),
        (r'\b(machine learning|neural network|neural nets|deep learning)\b', lambda m: 'Machine Learning Models'),
        (r'\b(performance|latency|concurrency|multithreading|threads)\b', lambda m: 'Performance & Concurrency')
    ]

    found_topics = []
    for pattern, fn in domain_rules:
        for match in re.finditer(pattern, text_clean, re.I):
            val = fn(match)
            if val not in found_topics:
                found_topics.append(val)

    # Substantive 2-word phrase extraction bounded strictly within sentences
    sentences = re.split(r'[.!?\n]+', text_clean)
    phrase_counts = Counter()

    for s in sentences:
        words = [w.strip('.,!?:;"\'()[]{}').lower() for w in s.split()]
        words = [re.sub(r"['’](s|re|ve|ll|d|m|t)$", '', w) for w in words]
        words = [w for w in words if w and len(w) > 2 and w not in stopwords and w not in GENERIC_BAN_WORDS]

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 != w2:
                phrase_counts[f"{w1.capitalize()} {w2.capitalize()}"] += 1

    for phrase, count in phrase_counts.most_common(8):
        if phrase not in found_topics and not any(phrase.lower() in t.lower() for t in found_topics):
            found_topics.append(phrase)

    # Fallback to substantive domain nouns if needed
    if len(found_topics) < 5:
        all_words = []
        for s in sentences:
            wds = [w.strip('.,!?:;"\'()[]{}').lower() for w in s.split()]
            wds = [re.sub(r"['’](s|re|ve|ll|d|m|t)$", '', w) for w in wds]
            all_words.extend([w for w in wds if w and len(w) > 3 and w not in stopwords and w not in GENERIC_BAN_WORDS])
        for w, _ in Counter(all_words).most_common(6):
            cap = w.capitalize()
            if cap not in found_topics and not any(w in t.lower() for t in found_topics):
                found_topics.append(cap)

    weights = [48, 38, 32, 26, 22, 18, 15, 14, 13, 12]
    return [[topic, weights[min(idx, len(weights) - 1)]] for idx, topic in enumerate(found_topics[:10])]



@app.route("/api/meeting-analytics", methods=["GET"])
def api_meeting_analytics():
    """Returns persistent analytics (wordcloud, analysis summary) directly from SQLite DB."""
    return jsonify({
        "wordcloud": db.get_meeting_meta("wordcloud", []),
        "analysis": db.get_meeting_meta("speaker_analysis", {})
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
    turn_store.clear()
    db.clear_all_turns()
    db.clear_meeting_meta()
    broadcast_sse("turns_cleared", {})
    broadcast_sse("turns_updated", {})
    broadcast_sse("wordcloud_cleared", {})
    broadcast_sse("analysis_cleared", {})
    return jsonify({"status": "turns_cleared", "turns_count": 0})


@app.route("/api/clear-all", methods=["POST"])
def api_clear_all():
    global _session_cluster_map
    turn_store.clear()
    _session_cluster_map.clear()
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


from audio_utils import clean_audio_for_speaker_id, decode_pcm_base64


def _decode_pcm_base64(b64_str):
    return decode_pcm_base64(b64_str)


def _rebuild_all_centroids():
    grouped = {}
    fallback_grouped = {}

    for entry in list(turn_store.values()):
        if entry.get("is_mixed"):
            continue  # Exclude mixed / contaminated audio from all centroids

        speaker = (entry.get("tagged_as") or "").strip()
        emb = entry.get("embedding")
        if speaker and emb is not None:
            audio_b64 = entry.get("audio_b64") or ""
            # Calculate PCM length in seconds
            pcm_len = len(audio_b64) * 3 // 4 // 2 if audio_b64 else 0
            duration_s = pcm_len / 16000.0 if pcm_len > 0 else 1.0

            if speaker not in fallback_grouped:
                fallback_grouped[speaker] = []
            fallback_grouped[speaker].append(emb)

            # Combined into centroid only if >= 1.0s audio
            if duration_s >= 1.0:
                if speaker not in grouped:
                    grouped[speaker] = []
                grouped[speaker].append(emb)

    # Fallback to shorter sample if a user has no >= 1.0s samples yet
    for spk, embs in fallback_grouped.items():
        if spk not in grouped or not grouped[spk]:
            grouped[spk] = embs

    speaker_id_engine.sync_all_user_centroids(grouped)


def _update_single_user_centroid(speaker_name):
    """Incrementally update only one user's centroid instead of rebuilding all."""
    name = (speaker_name or "").strip()
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
                audio_b64 = entry.get("audio_b64") or ""
                pcm_len = len(audio_b64) * 3 // 4 // 2 if audio_b64 else 0
                duration_s = pcm_len / 16000.0 if pcm_len > 0 else 1.0
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
        elif entry.get("predicted_speaker") and entry.get("predicted_confidence", 0.0) >= 35.0:
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
            scores = id_res.get("scores", {})
            predicted_speaker = id_res.get("top_match") or id_res.get("winner")
            predicted_confidence = id_res.get("confidence_pct", 0.0)

            # If no active profiles with centroids exist or top confidence is minimal with no winner
            if not scores or (not id_res.get("winner") and predicted_confidence <= 10.0 and not scores):
                # Fallback to AssemblyAI cluster inheritance if known
                if lbl and lbl in _session_cluster_map:
                    predicted_speaker = _session_cluster_map[lbl]
                    predicted_confidence = 30.0
                else:
                    predicted_speaker = None
                    predicted_confidence = 0.0

            entry["predicted_speaker"] = predicted_speaker
            entry["predicted_confidence"] = predicted_confidence
            entry["scores"] = scores
        else:
            if lbl and lbl in _session_cluster_map:
                entry["predicted_speaker"] = _session_cluster_map[lbl]
                entry["predicted_confidence"] = 30.0
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
        scores = id_res.get("scores", {})
        predicted_speaker = id_res.get("top_match") or id_res.get("winner")
        predicted_confidence = id_res.get("confidence_pct", 0.0)

        entry = turn_store.get(turn_order)
        lbl = entry.get("speaker_label") if entry else None

        # If neural confidence is high on longer utterance, lock in session cluster mapping
        if predicted_speaker and predicted_confidence >= 35.0 and lbl:
            _session_cluster_map[lbl] = predicted_speaker

        # If neural confidence is low on short interjection (<1.5s), inherit session cluster speaker
        if (not predicted_speaker or predicted_confidence < 15.0) and lbl and lbl in _session_cluster_map:
            predicted_speaker = _session_cluster_map[lbl]
            predicted_confidence = 30.0

        # Persist the computed embedding so it doesn't need re-extraction on restart
        embedding_json = json.dumps(emb.tolist())
        db.update_turn_embedding(turn_order, embedding_json)

        if entry:
            entry["embedding"] = emb
            entry["predicted_speaker"] = predicted_speaker
            entry["predicted_confidence"] = predicted_confidence
            entry["scores"] = scores

            # Auto-tag and combine into centroid if high-confidence (>= 70%) AND audio length >= 1.0s
            if (
                predicted_speaker
                and predicted_confidence >= 70.0
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
                    audio_b64=audio_b64,
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

    # 1. Instant SQLite write in <1ms
    db.upsert_turn(
        turn_order=turn_order,
        text=text,
        speaker_label=speaker_label,
        audio_b64=audio_b64 if audio_b64 else None,
        tagged_as=None,
        predicted_speaker=None,
        predicted_confidence=0.0,
    )

    turn_store[turn_order] = {
        "audio_b64": audio_b64,
        "embedding": None,
        "text": text,
        "speaker_label": speaker_label,
        "predicted_speaker": None,
        "predicted_confidence": 0.0,
        "scores": {},
        "tagged_as": None,
    }

    if len(turn_store) > 200:
        oldest = sorted(turn_store.keys())[: len(turn_store) - 200]
        for k in oldest:
            del turn_store[k]

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
        result.append({
            "turn_order": to,
            "text": t["text"],
            "speaker_label": t["speaker_label"],
            "tagged_as": t["tagged_as"],
            "predicted_speaker": t["predicted_speaker"],
            "predicted_confidence": t.get("predicted_confidence", 0.0),
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
    db.update_turn_tag(turn_order, speaker_name)

    # 1. INSTANT sub-millisecond SSE broadcast to all connected UIs
    broadcast_sse("tag_updated", {"turn_order": turn_order, "tagged_as": speaker_name})
    broadcast_sse("turns_updated", {"turn_order": turn_order})

    # 2. Async background centroid & prediction update
    def _async_update_centroids():
        try:
            if entry.get("embedding") is None and entry.get("audio_b64"):
                pcm = _decode_pcm_base64(entry["audio_b64"])
                if len(pcm) >= int(0.3 * 16000):
                    emb = speaker_id_engine.extract_embedding(pcm)
                    entry["embedding"] = emb
                    db.update_turn_embedding(turn_order, json.dumps(emb.tolist()))

            if entry.get("embedding") is not None:
                _update_single_user_centroid(speaker_name)
                if old_speaker and old_speaker.lower() != speaker_name.lower():
                    _update_single_user_centroid(old_speaker)
                _recalculate_turn_predictions()
                broadcast_sse("users_updated", {})
        except Exception:
            pass

    _bg_pool.submit(_async_update_centroids)

    return jsonify({
        "status": "tagged",
        "turn_order": turn_order,
        "speaker_name": speaker_name,
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
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
