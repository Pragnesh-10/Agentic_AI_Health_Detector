# Health Symptom Checker
Agentic AI health symptom checker built with IBM Granite on Watsonx.ai.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in IBM credentials
3. Place data files in `data/raw/`
4. Run `python preprocess.py`
5. Run `python build_vector_db.py`
6. Run `uvicorn main:app --reload`
7. Open `http://localhost:8000`

## IBM Credentials Setup
- Sign up at https://cloud.ibm.com (Lite tier, free)
- Create Watson Machine Learning instance
- Create a project at https://dataplatform.cloud.ibm.com
- Get API key from https://cloud.ibm.com/iam/apikeys
- Copy Project ID from project settings

## Architecture
5-agent sequential pipeline:
1. Language Detection Agent (IBM Granite)
2. Symptom Extractor Agent (IBM Granite)
3. RAG Retriever Agent (FAISS + sentence-transformers)
4. Urgency Classifier Agent (rule-based)
5. Recommendation Agent (IBM Granite + RAG context)

## Tech Stack
- LLM: IBM Granite 3.8B Instruct via Watsonx.ai
- Vector DB: FAISS
- Embeddings: all-MiniLM-L6-v2
- Backend: FastAPI
- Frontend: Vanilla HTML/CSS/JS
