from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.IndexFlatL2(384)
docs = []

def add_embedding(text):
    vec = model.encode([text])[0]
    index.add(np.array([vec]).astype("float32"))
    docs.append(text)

def search_embedding(query, k=5):
    q_vec = model.encode([query])[0]
    D, I = index.search(np.array([q_vec]).astype("float32"), k)

    return [docs[i] for i in I[0] if i < len(docs)]