from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pipeline import run
from config import logger
import os

app = FastAPI(title="Health Symptom Checker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

class SymptomRequest(BaseModel):
    message: str

class SymptomResponse(BaseModel):
    language_detected: str
    symptoms_detected: str
    possible_conditions: list
    urgency: dict
    recommendation: str
    sources_used: int
    disclaimer: str

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/check", response_model=SymptomResponse)
def check_symptoms(request: SymptomRequest):
    if not request.message or len(request.message.strip()) < 5:
        raise HTTPException(
            status_code=400, 
            detail="Please describe your symptoms in more detail."
        )
    if len(request.message) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Input too long. Please keep it under 1000 characters."
        )
    try:
        result = run(request.message)
        return result
    except Exception as e:
        logger.exception("Analysis failed due to exception")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health")
def health_check():
    try:
        from agents.rag_retriever import _load
        _load()
        return {"status": "ok", "message": "Health Symptom Checker is running natively and models are accessible."}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service Unavailable: Downstream dependencies are failing.")
