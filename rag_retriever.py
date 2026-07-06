import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index('vector_db/index.faiss')
with open('vector_db/chunks_meta.pkl', 'rb') as f:
    chunks = pickle.load(f)

encoder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query: str, top_k: int = 5) -> list:
    query_vec = encoder.encode([query]).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    results = [chunks[i]['text'] for i in indices[0] if i < len(chunks)]
    return results