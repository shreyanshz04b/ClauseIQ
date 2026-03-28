import re
import json
import math
from app.db import get_conn
from rank_bm25 import BM25Okapi


def normalize_query(q):
    q = q.lower()

    mapping = {
        "banka": "bank",
        "unfreze": "unfreeze",
        "account band": "bank account frozen",
        "paise nahi nikal": "cannot withdraw money",
        "kya karu": "what should i do",
        "kaise": "how"
    }

    for k, v in mapping.items():
        q = q.replace(k, v)

    return q


def expand_query(q):
    extra = []

    if "unfreeze" in q:
        extra += ["unblock account", "kyc update", "bank freeze reason"]

    if "bank" in q:
        extra += ["rbi rules", "account restriction", "kyc compliance"]

    return q + " " + " ".join(extra)


def keywords(text):
    return list(set(re.findall(r'\b[a-z]{3,}\b', text.lower())))


def extract_laws(text):
    text = text.lower()
    laws = set()

    patterns = [
        r'ipc\s*\d+',
        r'crpc\s*\d+',
        r'section\s*\d+',
        r'rbi'
    ]

    for p in patterns:
        for m in re.findall(p, text):
            laws.add(m.upper())

    if "fraud" in text:
        laws.add("IPC 420")

    return list(laws)


def chunk(text, size=300):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) < size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current)

    return chunks


def add_doc(pages, source="uploaded"):
    conn = get_conn()
    cur = conn.cursor()

    total = 0

    for p in pages:
        page_no = p["page"]
        text = p["text"]

        parts = chunk(text)

        for c in parts:
            cur.execute(
                "INSERT INTO docs(content,keywords,page,source) VALUES(?,?,?,?)",
                (
                    c,
                    json.dumps({
                        "kw": keywords(c),
                        "laws": extract_laws(c)
                    }),
                    page_no,
                    source
                )
            )
            total += 1

    conn.commit()
    conn.close()
    return total


def vectorize(text):
    vec = {}
    for w in keywords(text):
        vec[w] = vec.get(w, 0) + 1
    return vec


def cosine(v1, v2):
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in v1)
    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))

    if mag1 * mag2 == 0:
        return 0

    return dot / (mag1 * mag2)


def retrieve(query):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT content,page FROM docs")
    rows = cur.fetchall()
    conn.close()

    docs = [{"text": r[0], "page": r[1]} for r in rows]

    if not docs:
        return []

    texts = [d["text"] for d in docs]

    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.split())

    q_vec = vectorize(query)

    results = []

    for i, d in enumerate(docs):
        sem = cosine(q_vec, vectorize(d["text"]))
        score = 0.6 * bm25_scores[i] + 0.4 * sem

        if score > 0.2:
            results.append({
                "score": score,
                "text": d["text"],
                "page": d["page"]
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]


def rerank(query, contexts):
    ranked = []

    for c in contexts:
        score = c["score"]

        if "freeze" in query and "freeze" in c["text"].lower():
            score += 1

        if "kyc" in c["text"].lower():
            score += 0.5

        ranked.append({
            "score": score,
            "text": c["text"],
            "page": c["page"]
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def confidence_score(contexts, answer):
    if not contexts:
        return 0.2

    joined = " ".join(contexts).lower()
    ans = answer.lower()

    overlap = len(set(ans.split()) & set(joined.split()))

    if overlap > 40:
        return 0.9
    elif overlap > 20:
        return 0.75
    elif overlap > 10:
        return 0.6
    else:
        return 0.4


def highlight_citations(answer, contexts):
    citations = []

    ans_words = set(answer.lower().split())

    for c in contexts:
        lines = re.split(r'[.?!]', c["text"])

        for line in lines:
            words = set(line.lower().split())
            overlap = len(ans_words & words)

            if overlap > 6 and len(line) > 40:
                citations.append({
                    "text": line.strip(),
                    "page": c["page"]
                })

    return citations[:5]


def store_qa(query, answer):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO docs(content,keywords,page,source) VALUES(?,?,?,?)",
        (
            query + " " + answer,
            json.dumps({
                "kw": keywords(query),
                "laws": extract_laws(query)
            }),
            0,
            "qa_memory"
        )
    )

    conn.commit()
    conn.close()