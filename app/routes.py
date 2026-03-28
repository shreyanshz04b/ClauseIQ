from flask import Blueprint, render_template, request, jsonify
import os, requests, re, json
from config import UPLOAD_FOLDER, OPENROUTER_API_KEY
from app.db import get_conn

from app.rag_engine import (
    add_doc, retrieve, extract_laws,
    normalize_query, store_qa,
    rerank, confidence_score, highlight_citations
)

from app.utils_extractor import extract_text_with_pages

bp = Blueprint("main", __name__)

MODELS = [
    "mistralai/mistral-7b-instruct",
    "openchat/openchat-7b"
]


def sanitize(text):
    return re.sub(r'[^a-zA-Z0-9 .,?]', '', text)[:500]


def safe_llm_call(prompt):
    for model in MODELS:
        for _ in range(2):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    },
                    timeout=15
                )

                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]

            except:
                continue

    return None


def understand_query(query):
    prompt = f"""
Convert user query into JSON.

Query: {query}

Return ONLY JSON:
{{
 "clean_query": "...",
 "intent": "summary/explain/procedure/legal/advice"
}}
"""

    res = safe_llm_call(prompt)

    try:
        res = res.strip()
        start = res.find("{")
        end = res.rfind("}") + 1
        res = res[start:end]

        parsed = json.loads(res)

        clean_query = parsed.get("clean_query", query)
        intent = parsed.get("intent", "explain")

    except:
        clean_query = query
        intent = "explain"

    q = query.lower()

    if any(x in q for x in ["kya hai", "kya keh", "samjha", "meaning"]):
        intent = "explain"

    if any(x in q for x in ["kaise", "how", "process"]):
        intent = "procedure"

    if any(x in q for x in ["summary", "summarize", "short"]):
        intent = "summary"

    return {
        "clean_query": clean_query,
        "intent": intent
    }


def generate(context_texts, original_query, clean_query, intent):
    context_text = "\n\n".join(context_texts)
    laws = extract_laws(clean_query)
    law_text = ", ".join(laws) if laws else "Relevant Indian Law"

    if not context_text.strip():
        return "No relevant information found in document."

    prompt = f"""
You are a senior Indian legal AI.

STRICT:
- Use ONLY context
- Answer in SAME language as user
- No outside knowledge

INTENT: {intent}

FORMAT:

Summary:
Key Points:
Legal Meaning:
Action Steps:

Relevant Laws:
{law_text}

Context:
{context_text}

Query:
{original_query}
"""

    return safe_llm_call(prompt)


@bp.route('/')
def home():
    return render_template('index.html')


@bp.route('/legal-ai')
def legal_ai():
    return render_template('legal_ai.html')


@bp.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "no file"})

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        pages = extract_text_with_pages(path)

        if not pages:
            return jsonify({"error": "no readable content"})

        chunks = add_doc(pages, source=file.filename)

        preview = ""
        if pages:
            preview = pages[0]["text"][:500]

        print("\nEXTRACTED SAMPLE:\n", preview)

        return jsonify({
            "message": "stored",
            "chunks": chunks,
            "preview": preview
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"error": "upload failed"})


@bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"answer": "enter query"})

        query = sanitize(query)

        understood = understand_query(query)
        clean_query = normalize_query(understood["clean_query"])
        intent = understood["intent"]

        context_objs = retrieve(clean_query)
        context_objs = rerank(clean_query, context_objs)

        if not context_objs:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("SELECT content, page FROM docs ORDER BY id DESC LIMIT 10")
            rows = cur.fetchall()
            conn.close()

            context_objs = [{"text": r[0], "page": r[1]} for r in rows]

        context_texts = [c["text"] for c in context_objs]

        answer = generate(context_texts, query, clean_query, intent)

        if not answer:
            answer = "Not found in document."

        conf = confidence_score(context_texts, answer)
        cites = highlight_citations(answer, context_objs)

        print("\nUSER:", query)
        print("\nCLEAN:", clean_query)
        print("\nINTENT:", intent)
        print("\nCONF:", conf)

        store_qa(clean_query, answer)

        return jsonify({
            "answer": answer,
            "confidence": conf,
            "citations": cites
        })

    except Exception as e:
        print("ASK ERROR:", e)
        return jsonify({"answer": "server error"})