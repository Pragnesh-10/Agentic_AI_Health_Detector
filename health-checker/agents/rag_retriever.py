import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

_index = None
_chunks = None
_encoder = None

def _load():
    global _index, _chunks, _encoder
    if _index is None:
        _index = faiss.read_index("vector_db/index.faiss")
        with open("vector_db/chunks_meta.pkl", "rb") as f:
            _chunks = pickle.load(f)
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query: str, top_k: int = 5) -> list:
    _load()
    query_vec = _encoder.encode([query]).astype(np.float32)
    distances, indices = _index.search(query_vec, top_k)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i < len(_chunks):
            results.append({
                "text": _chunks[i]['text'],
                "disease": _chunks[i].get('disease', ''),
                "score": float(dist)
            })
    return results
