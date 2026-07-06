import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load chunks
with open('data/processed/chunks.json') as f:
    chunks = json.load(f)

texts = [c['text'] for c in chunks]

print("Embedding chunks... (takes 2-5 mins first time)")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

# Save index and metadata
faiss.write_index(index, 'vector_db/index.faiss')
with open('vector_db/chunks_meta.pkl', 'wb') as f:
    pickle.dump(chunks, f)

print("Vector DB built and saved.")