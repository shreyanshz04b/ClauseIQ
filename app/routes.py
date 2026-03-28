from flask import Blueprint, render_template, request, jsonify
import os, json, re, traceback, datetime, csv
from config import UPLOAD_FOLDER, GROQ_API_KEY
from app.db import get_conn
from app.rag_engine import add_doc, retrieve, rerank, confidence_score, highlight_citations, store_qa, extract_laws
from app.utils_extractor import extract_text_with_pages
from groq import Groq

bp = Blueprint("main", __name__)
client = Groq(api_key=GROQ_API_KEY)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sanitize(text):
    return text.strip()[:500]


def detect_lang(text):
    if re.search(r'[अ-ह]', text):
        return "hindi"
    if any(w in text.lower() for w in ["kya", "kaise", "karu", "hai"]):
        return "hinglish"
    return "english"


def groq_call(prompt):
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    except:
        return None


def extract_entities(text):
    accused = re.findall(r'(?:accused|petitioner|respondent)[\s:,-]+([A-Z][A-Za-z ]{3,40})', text)
    fir = re.findall(r'fir\s*(?:no\.?|number)?[:\s-]*([0-9/\\-]{3,20})', text, re.I)
    court = re.findall(r'(high court|supreme court|district court)[^.,\n]{0,40}', text, re.I)
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)

    return {
        "accused": list(set(accused))[:3] or ["Not found"],
        "fir": list(set(fir))[:2] or ["Not found"],
        "court": list(set(court))[:2] or ["Not found"],
        "dates": list(set(dates))[:3] or ["Not found"]
    }


def load_all_laws():
    datasets = {}

    files = [
        "ipc.json", "crpc.json", "cpc.json",
        "iea.json", "hma.json", "ida.json", "nia.json"
    ]

    for f in files:
        try:
            with open(f"data/{f}", encoding="utf-8") as fp:
                datasets[f] = json.load(fp)
        except:
            datasets[f] = []

    bns = {}
    try:
        with open("data/bns_sections.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                bns[str(r.get("section"))] = r.get("description")
    except:
        pass

    return datasets, bns


LAW_DATA, BNS_DATA = load_all_laws()


def get_law_text(section):
    num = re.findall(r'\d+', section)
    if not num:
        return None
    num = num[0]

    for dataset in LAW_DATA.values():
        for s in dataset:
            if str(s.get("section") or s.get("Section")) == num:
                return s.get("section_desc") or s.get("description")

    if num in BNS_DATA:
        return BNS_DATA[num]

    return None


def generate(contexts, query):
    lang = detect_lang(query)

    context = "\n\n".join([f"[Page {c['page']}] {c['text']}" for c in contexts[:6]])

    laws = extract_laws(context)
    entities = extract_entities(context)

    law_texts = []
    for l in laws:
        txt = get_law_text(l)
        if txt:
            law_texts.append(f"{l}: {txt[:300]}")

    base_prompt = f"""
LEGAL FORENSIC ENGINE

QUERY:
{query}

CONTEXT:
{context}

LAW:
{" ".join(law_texts)}

OUTPUT:

Case Summary:
Facts:
Entities:
Sections:
Legal Reasoning:
Legal Strategy:
Next Step:
Judgment Prediction:
Confidence:
"""

    res = groq_call(base_prompt)

    if res and len(res) > 120:
        return translate_output(res, lang)

    fallback = f"""
Explain clearly what is happening and what user should do.

Query:
{query}

Context:
{context}

Give practical answer with law and next steps.
"""

    res2 = groq_call(fallback)

    if res2:
        return translate_output(res2, lang)

    return translate_output("Relevant legal content found but structured extraction failed.", lang)


def translate_output(text, lang):
    if lang == "english":
        return text

    prompt = f"Translate into {lang} but keep legal terms same:\n{text}"
    res = groq_call(prompt)

    return res if res else text


def save_output(query, answer):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{OUTPUT_DIR}/response_{ts}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"QUERY:\n{query}\n\nRESPONSE:\n{answer}")

    return path


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

        return jsonify({"message": "stored", "chunks": chunks})

    except:
        return jsonify({"error": "upload failed"})


@bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = sanitize(data.get("query", ""))

        if not query:
            return jsonify({"answer": "Enter query"})

        q = query + " ipc crpc law section court india"

        contexts = retrieve(q)
        contexts = rerank(q, contexts)

        if not contexts:
            return jsonify({"answer": "No relevant data found", "confidence": 0.2})

        filtered = [c for c in contexts if len(c["text"]) > 80]

        if len(filtered) < 2:
            filtered = contexts[:5]

        answer = generate(filtered, query)

        conf = confidence_score(filtered, answer)
        cites = highlight_citations(answer, filtered)
        file_path = save_output(query, answer)

        try:
            store_qa(query, answer)
        except:
            pass

        return jsonify({
            "answer": answer,
            "confidence": conf,
            "citations": cites,
            "file": file_path,
            "credibility": {
                "score": conf,
                "chunks_used": len(filtered)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"answer": "Server error"})