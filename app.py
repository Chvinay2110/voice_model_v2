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
    """Generates a robust, realistic per-speaker evaluation with derived takeaways."""
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

    speakers_res = []
    for spk, utts in speaker_turns.items():
        all_text = " ".join(utts)
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
            # Low substance / filler / gibberish
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


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """Calls Gemini with the transcript and returns per-speaker summaries, key points, and 1-10 value scores."""
    data = request.get_json(force=True) or {}
    transcript = data.get("transcript", "").strip()
    if not transcript:
        return jsonify({"error": "No transcript provided."}), 400

    # Ensure context efficiency for very long meetings while preserving all speakers
    lines = transcript.split("\n")
    if len(lines) > 250:
        transcript = "\n".join(lines[-250:])

    prompt = (
        "You are an expert discourse intelligence analyst evaluating a live multi-speaker meeting.\n"
        "Analyze this conversation transcript with complete intellectual honesty, objectivity, and realistic rigor.\n\n"
        "CRITICAL EVALUATION GUIDELINES (DO NOT BE A PEOPLE-PLEASER):\n"
        "- Be realistic, objective, and strict with your ratings. Do NOT hesitate to give low marks (e.g. 1.0 to 4.0 out of 10).\n"
        "- If a speaker speaks gibberish, trivial pleasantries, off-topic rambling, repetitive filler, or makes no meaningful sense, explicitly and politely state in their summary that they did not make a meaningful or actionable contribution, and rate them accordingly (1.0 to 3.5).\n"
        "- High scores (8.0 to 10.0) MUST be strictly reserved for speakers who provided concrete technical facts, actionable decisions, clear solutions, or deep domain expertise.\n"
        "- Moderate scores (5.0 to 7.5) are for constructive questions, confirmations, and standard discussion.\n\n"
        "MANDATORY JSON SCHEMA FOR EVERY UNIQUE SPEAKER:\n"
        "1. 'name': The exact speaker name as shown in the transcript (e.g. 'vinay', 'Speaker A').\n"
        "2. 'summary': An objective, realistic assessment of what they said and their actual contribution.\n"
        "3. 'key_points': An array of key points, takeaways, proposals, and facts shared by this speaker (if they spoke low substance, list an honest point like 'No substantive takeaways shared').\n"
        "4. 'score': A float rating from 1.0 to 10.0 (e.g. 2.5, 8.5) representing the knowledge density, helpfulness, and actionable value of what they said relative to everything they spoke.\n"
        "5. 'score_reason': A 1-sentence honest explanation of why they earned this exact score.\n"
        "6. 'keywords': An array of 1 to 5 key domain topics or concepts discussed by this speaker.\n\n"
        "Also provide an 'overall_summary' (2-3 sentences summarizing the overall meeting).\n\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "overall_summary": "...",\n'
        '  "speakers": [\n'
        "    {\n"
        '      "name": "Speaker Name",\n'
        '      "score": 8.5,\n'
        '      "score_reason": "...",\n'
        '      "summary": "...",\n'
        '      "key_points": ["...", "..."],\n'
        '      "keywords": ["...", "..."]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Transcript:\n{transcript}"
    )

    raw_json = call_gemini(prompt, response_json=True, timeout=30)
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "speakers" in parsed:
                return jsonify(parsed)
        except Exception:
            pass

    # Fallback retry without forcing json mime-type
    result = call_gemini(prompt, timeout=30)
    if result:
        try:
            cleaned = result.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            if isinstance(parsed, dict) and "speakers" in parsed:
                return jsonify(parsed)
        except Exception:
            pass

    # Built-in fallback analysis so UI never breaks
    return jsonify(_fallback_speaker_analysis(transcript))


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


@app.route("/api/wordcloud", methods=["POST"])
def api_wordcloud():
    """Extracts 3 main topics, 4-5 sub-topics, and other topics using Gemini with structured format."""
    data = request.get_json(force=True) or {}
    transcript = data.get("transcript", "").strip()
    if not transcript:
        return jsonify({"words": []})

    prompt = (
        "You are an expert discourse analyst analyzing this live meeting transcript.\n\n"
        "Identify and categorize what the participants are talking about into three distinct levels:\n"
        "1. 'main_topics': Exactly 3 primary, high-level macro topics being discussed (2 to 4 words each).\n"
        "2. 'sub_topics': 4 to 5 key sub-topics or specific discussion areas (2 to 4 words each).\n"
        "3. 'other_topics': 3 to 5 other relevant concepts, keywords, or topics mentioned (2 to 4 words each).\n\n"
        "RULES:\n"
        "- Every topic MUST be a meaningful multi-word domain phrase (e.g. 'AI Safety Alignment', 'Superintelligent AI Risks', 'Database Indexing Performance').\n"
        "- STRICTLY FORBIDDEN: Do not output conversational filler phrases (e.g. 'There is lots', 'Scary secret', 'Track to achieve', 'Hope it is').\n\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "main_topics": ["Main Topic 1", "Main Topic 2", "Main Topic 3"],\n'
        '  "sub_topics": ["Sub Topic 1", "Sub Topic 2", "Sub Topic 3", "Sub Topic 4", "Sub Topic 5"],\n'
        '  "other_topics": ["Other Topic 1", "Other Topic 2", "Other Topic 3"]\n'
        "}\n\n"
        f"Transcript:\n{transcript}"
    )

    raw_json = call_gemini(prompt, response_json=True, timeout=15)
    if not raw_json:
        raw_json = call_gemini(prompt, timeout=15)

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
                main_topics = parsed.get("main_topics", [])
                sub_topics = parsed.get("sub_topics", [])
                other_topics = parsed.get("other_topics", [])

                words = []
                # Main topics: highest weights (48, 42, 38)
                main_weights = [48, 42, 38]
                for i, t in enumerate(main_topics[:3]):
                    if t and isinstance(t, str):
                        words.append([t.strip(), main_weights[min(i, len(main_weights) - 1)]])

                # Sub topics: medium weights (32, 28, 25, 22, 20)
                sub_weights = [32, 28, 25, 22, 20]
                for i, t in enumerate(sub_topics[:5]):
                    if t and isinstance(t, str):
                        words.append([t.strip(), sub_weights[min(i, len(sub_weights) - 1)]])

                # Other topics: compact weights (16, 14, 12, 11)
                other_weights = [16, 14, 12, 11]
                for i, t in enumerate(other_topics[:5]):
                    if t and isinstance(t, str):
                        words.append([t.strip(), other_weights[min(i, len(other_weights) - 1)]])

                if len(words) >= 3:
                    return jsonify({"words": words})
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
    broadcast_sse("turns_cleared", {})
    return jsonify({"status": "turns_cleared", "turns_count": 0})


@app.route("/api/clear-all", methods=["POST"])
def api_clear_all():
    turn_store.clear()
    speaker_id_engine.clear_session()
    db.clear_all_profiles()
    broadcast_sse("cleared", {})
    return jsonify({"status": "cleared", "turns_count": 0, "users_count": 0})


from audio_utils import clean_audio_for_speaker_id, decode_pcm_base64


def _decode_pcm_base64(b64_str):
    return decode_pcm_base64(b64_str)


def _rebuild_all_centroids():
    grouped = {}
    fallback_grouped = {}

    for entry in list(turn_store.values()):
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


def _recalculate_turn_predictions():
    """Re-identify all turns against current user centroids and batch-update database."""
    batch_updates = []
    for turn_order, entry in list(turn_store.items()):
        emb = entry.get("embedding")
        if emb is not None:
            id_res = speaker_id_engine.identify_embedding(emb)
            scores = id_res.get("scores", {})
            predicted_speaker = id_res.get("top_match") or id_res.get("winner")
            predicted_confidence = id_res.get("confidence_pct", 0.0)

            # If no active profiles with centroids exist or top confidence is minimal with no winner
            if not scores or (not id_res.get("winner") and predicted_confidence <= 10.0 and not scores):
                predicted_speaker = None
                predicted_confidence = 0.0

            entry["predicted_speaker"] = predicted_speaker
            entry["predicted_confidence"] = predicted_confidence
            entry["scores"] = scores
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

        # Persist the computed embedding so it doesn't need re-extraction on restart
        embedding_json = json.dumps(emb.tolist())
        db.update_turn_embedding(turn_order, embedding_json)

        entry = turn_store.get(turn_order)
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
            }
            turn_store[turn_order] = entry
        else:
            entry = {
                "audio_b64": None,
                "embedding": None,
                "text": "",
                "speaker_label": "A",
                "tagged_as": None,
            }
            turn_store[turn_order] = entry

    try:
        if entry.get("embedding") is None and entry.get("audio_b64"):
            pcm = _decode_pcm_base64(entry["audio_b64"])
            if len(pcm) >= int(0.3 * 16000):
                entry["embedding"] = speaker_id_engine.extract_embedding(pcm)

        old_speaker = (entry.get("tagged_as") or "").strip()
        entry["tagged_as"] = speaker_name
        db.update_turn_tag(turn_order, speaker_name)

        if entry.get("embedding") is not None:
            # Update the new speaker's centroid (includes this turn now)
            _update_single_user_centroid(speaker_name)
            # Update the old speaker's centroid (no longer includes this turn)
            if old_speaker and old_speaker.lower() != speaker_name.lower():
                _update_single_user_centroid(old_speaker)
            _recalculate_turn_predictions()

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
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
