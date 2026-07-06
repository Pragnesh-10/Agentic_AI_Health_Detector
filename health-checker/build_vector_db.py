import json
import numpy as np
import faiss
import pickle
import os
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import Credentials
from config import IBM_API_KEY, IBM_PROJECT_ID, IBM_URL

os.makedirs("vector_db", exist_ok=True)

with open("data/processed/chunks.json") as f:
    chunks = json.load(f)

texts = [c['text'] for c in chunks]
print(f"Embedding {len(texts)} chunks using WatsonX...")

credentials = Credentials(url=IBM_URL, api_key=IBM_API_KEY)
embedding_model = Embeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    credentials=credentials,
    project_id=IBM_PROJECT_ID
)

# Batch embedding to avoid API limits
batch_size = 50
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    print(f"Embedding batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")
    vectors = embedding_model.embed_documents(batch)
    all_embeddings.extend(vectors)

embeddings = np.array(all_embeddings).astype(np.float32)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "vector_db/index.faiss")
with open("vector_db/chunks_meta.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"Vector DB saved. Dimension: {dimension}, Total vectors: {index.ntotal}")
