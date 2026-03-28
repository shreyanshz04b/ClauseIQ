import re
import json
import math
from collections import defaultdict
from app.db import get_conn
from rank_bm25 import BM25Okapi


LEGAL_TERMS = [
    "ipc", "crpc", "section", "court", "judge",
    "police", "offence", "complaint", "evidence",
    "investigation", "accused", "fir", "petition",
    "order", "judgment", "bns", "act"
]


def keywords(text):
    return re.findall(r'\b[a-z0-9]{3,}\b', text.lower())


def extract_laws(text):
    text = text.lower()
    laws = set()

    patterns = [
        r'ipc\s*\d+',
        r'crpc\s*\d+',
        r'bns\s*\d+',
        r'section\s*\d+',
        r'article\s*\d+'
    ]

    for p in patterns:
        laws.update([m.upper() for m in re.findall(p, text)])

    if "fraud" in text:
        laws.add("IPC 420")

    return list(laws)


def detect_intent(query):
    q = query.lower()

    if any(x in q for x in ["summary", "summarize"]):
        return "summary"
    if any(x in q for x in ["section", "law", "ipc", "crpc"]):
        return "law_lookup"
    if any(x in q for x in ["what to do", "next step", "action"]):
        return "strategy"
    if any(x in q for x in ["case", "accused", "fir"]):
        return "case"

    return "general"


def legal_density(text):
    return sum(1 for k in LEGAL_TERMS if k in text.lower())


def entity_score(text):
    t = text.lower()
    score = 0

    if "fir" in t:
        score += 3
    if "accused" in t:
        score += 3
    if "section" in t:
        score += 2
    if "court" in t:
        score += 1

    return score


def smart_chunk(text, size=280):
    sents = re.split(r'(?<=[.!?]) +', text)
    chunks, cur = [], ""

    for s in sents:
        if len(cur) + len(s) < size:
            cur += " " + s
        else:
            if legal_density(cur) > 1:
                chunks.append(cur.strip())
            cur = s

    if cur and legal_density(cur) > 1:
        chunks.append(cur.strip())

    return chunks


def expand_query(query):
    q = query.lower()

    mapping = {
        "fraud": ["ipc 420", "cheating"],
        "police": ["fir", "complaint"],
        "court": ["judge", "hearing"],
        "case": ["evidence", "offence"],
        "bank": ["freeze", "rbi"]
    }

    extra = []
    for k, v in mapping.items():
        if k in q:
            extra += v

    return q + " " + " ".join(extra)


def add_doc(pages, source="uploaded"):
    conn = get_conn()
    cur = conn.cursor()

    total = 0

    for p in pages:
        page = p["page"]
        chunks = smart_chunk(p["text"])

        for c in chunks:
            cur.execute(
                "INSERT INTO docs(content,keywords,page,source) VALUES(?,?,?,?)",
                (
                    c,
                    json.dumps({
                        "kw": keywords(c),
                        "laws": extract_laws(c),
                        "density": legal_density(c),
                        "entity": entity_score(c)
                    }),
                    page,
                    source
                )
            )
            total += 1

    conn.commit()
    conn.close()
    return total


def vectorize(text):
    v = defaultdict(int)
    for w in keywords(text):
        v[w] += 1
    return v


def cosine(a, b):
    dot = sum(a[k] * b.get(k, 0) for k in a)
    ma = math.sqrt(sum(x * x for x in a.values()))
    mb = math.sqrt(sum(x * x for x in b.values()))
    return dot / (ma * mb) if ma and mb else 0


def retrieve(query):
    intent = detect_intent(query)
    query = expand_query(query)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT content,page,keywords FROM docs")
    rows = cur.fetchall()
    conn.close()

    docs = []
    for r in rows:
        meta = json.loads(r[2]) if r[2] else {}
        docs.append({
            "text": r[0],
            "page": r[1],
            "meta": meta
        })

    if not docs:
        return []

    tokenized = [keywords(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    bm_scores = bm25.get_scores(keywords(query))

    q_vec = vectorize(query)

    results = []

    for i, d in enumerate(docs):
        text = d["text"]
        meta = d["meta"]

        sem = cosine(q_vec, vectorize(text))
        bm = bm_scores[i]

        density = meta.get("density", 0)
        entity = meta.get("entity", 0)
        law_bonus = len(meta.get("laws", []))

        score = (
            0.45 * bm +
            0.25 * sem +
            0.15 * density +
            0.10 * entity +
            0.05 * law_bonus
        )

        if intent == "law_lookup":
            score += law_bonus * 2

        if intent == "case":
            score += entity * 2

        if density < 1:
            continue

        results.append({
            "score": score,
            "text": text,
            "page": d["page"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return diversify(results[:20])


def diversify(results):
    final = []
    seen = set()

    for r in results:
        key = r["text"][:140]
        if key in seen:
            continue

        seen.add(key)
        final.append(r)

        if len(final) == 10:
            break

    return final


def rerank(query, contexts):
    q = query.lower()
    ranked = []

    for c in contexts:
        score = c["score"]
        text = c["text"].lower()

        if "section" in text or "ipc" in text:
            score += 2

        if "fir" in text or "accused" in text:
            score += 2

        if "evidence" in text:
            score += 1

        if "judgment" in text:
            score += 1.5

        if any(x in q for x in ["crime", "case"]):
            score += 0.5

        ranked.append({
            "score": score,
            "text": c["text"],
            "page": c["page"]
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:6]


def confidence_score(contexts, answer):
    joined = " ".join([c["text"] for c in contexts]).lower()
    ans = answer.lower()

    overlap = len(set(ans.split()) & set(joined.split()))

    if overlap > 80:
        return 0.9
    if overlap > 50:
        return 0.75
    if overlap > 30:
        return 0.6
    return 0.4


def highlight_citations(answer, contexts):
    citations = []
    ans_words = set(answer.lower().split())

    for c in contexts:
        for line in re.split(r'[.?!]', c["text"]):
            if len(line) < 60:
                continue

            overlap = len(ans_words & set(line.lower().split()))

            if overlap > 6:
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