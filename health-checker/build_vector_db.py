import json
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

os.makedirs("vector_db", exist_ok=True)

with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

texts = [c['text'] for c in chunks]
print(f"Embedding {len(texts)} chunks...")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

faiss.write_index(index, "vector_db/index.faiss")
with open("vector_db/chunks_meta.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"Vector DB saved. Dimension: {dimension}, Total vectors: {index.ntotal}")
