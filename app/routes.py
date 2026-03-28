from flask import Blueprint, render_template, request, jsonify
import os, json, traceback
from config import UPLOAD_FOLDER, GROQ_API_KEY
from app.db import get_conn

from app.rag_engine import (
    add_doc, retrieve,
    rerank, confidence_score,
    highlight_citations, store_qa
)

from app.utils_extractor import extract_text_with_pages
from groq import Groq

bp = Blueprint("main", __name__)

client = Groq(api_key=GROQ_API_KEY)


def sanitize(text):
    return text.strip()[:500]


def groq_call(prompt):
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        print("GROQ ERROR:", e)
        return None


def understand_query(query):
    prompt = f"""
Analyze user query and return JSON:

{{
 "mode": "document" or "search",
 "intent": "summary/explain/procedure/legal/advice",
 "query": "clean English version"
}}

Query: {query}
"""

    res = groq_call(prompt)

    try:
        parsed = json.loads(res)
        return {
            "mode": parsed.get("mode", "search"),
            "intent": parsed.get("intent", "explain"),
            "query": parsed.get("query", query)
        }
    except:
        return {"mode": "search", "intent": "explain", "query": query}


def generate(context_texts, user_query, intent):
    if not context_texts:
        return "No relevant content found in document."

    context = "\n\n".join(context_texts[:8])

    prompt = f"""
You are a professional Indian legal assistant.

STRICT:
- Answer ONLY from context
- DO NOT hallucinate
- Answer in same language as user (Hindi/Hinglish/English)
- Make answer clear and human-friendly

User Query:
{user_query}

Context:
{context}

Format:

Summary:
Key Points:
Legal Meaning:
Action Steps:
"""

    res = groq_call(prompt)

    if not res or len(res) < 20:
        return "LLM could not generate answer from document."

    return res


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

        return jsonify({
            "message": "stored",
            "chunks": chunks,
            "preview": pages[0]["text"][:500]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "upload failed"})


@bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = sanitize(data.get("query", ""))

        if not query:
            return jsonify({"answer": "Enter query"})

        q = understand_query(query)

        mode = q["mode"]
        clean_query = q["query"]
        intent = q["intent"]

        if mode == "document":
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("SELECT content, page FROM docs ORDER BY id DESC LIMIT 20")
            rows = cur.fetchall()
            conn.close()

            context_objs = [{"text": r[0], "page": r[1], "score": 1} for r in rows]

        else:
            combined_query = query + " " + clean_query
            context_objs = retrieve(combined_query)
            context_objs = rerank(combined_query, context_objs)

        if not context_objs:
            return jsonify({"answer": "No relevant data found in document"})

        context_texts = [c["text"] for c in context_objs]

        answer = generate(context_texts, query, intent)

        conf = confidence_score(context_objs, answer)
        cites = highlight_citations(answer, context_objs)

        try:
            store_qa(clean_query, answer)
        except:
            pass

        return jsonify({
            "answer": answer,
            "confidence": conf,
            "citations": cites
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": str(e)})