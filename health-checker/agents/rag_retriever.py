import faiss
import pickle
import numpy as np
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import Credentials
from config import IBM_API_KEY, IBM_PROJECT_ID, IBM_URL, IBM_EMBEDDING_MODEL, logger

_index = None
_chunks = None
_encoder = None

def _load():
    global _index, _chunks, _encoder
    if _index is None:
        _index = faiss.read_index("vector_db/index.faiss")
        with open("vector_db/chunks_meta.pkl", "rb") as f:
            _chunks = pickle.load(f)
        
        credentials = Credentials(url=IBM_URL, api_key=IBM_API_KEY)
        _encoder = Embeddings(
            model_id=IBM_EMBEDDING_MODEL,
            credentials=credentials,
            project_id=IBM_PROJECT_ID
        )

def retrieve(query: str, top_k: int = 5) -> list:
    try:
        _load()
        vectors = _encoder.embed_documents([query])
        query_vec = np.array(vectors).astype(np.float32)
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
    except Exception as e:
        logger.exception(f"Warning: RAG retrieval failed: {e}. Falling back to general knowledge.")
        return []
