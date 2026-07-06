from agents.rag_retriever import retrieve
from agents.granite_agent import (
    extract_symptoms, 
    generate_recommendation
)

def run(user_input: str) -> dict:
    # Agent 1 & 2 & 3: Detect language, extract symptoms, and translate to English in one pass
    language, symptoms_en = extract_symptoms(user_input)
    
    # Agent 4: Retrieve relevant medical context (using English symptoms)
    retrieved = retrieve(symptoms_en, top_k=5)
    context = "\n".join([r['text'] for r in retrieved[:3]])
    possible_diseases = list(set([r['disease'] for r in retrieved if r['disease']]))[:5]
    
    # Agent 5 & 6 & 7: Assess urgency, generate recommendation, and translate to user's language in one pass
    recommendation, urgency = generate_recommendation(symptoms_en, context, language)
    
    return {
        "language_detected": language,
        "symptoms_detected": symptoms_en,
        "possible_conditions": possible_diseases,
        "urgency": urgency,
        "recommendation": recommendation,
        "sources_used": len(retrieved),
        "disclaimer": "⚠️ This is educational information only, not a medical diagnosis. Always consult a licensed physician for proper medical advice."
    }

