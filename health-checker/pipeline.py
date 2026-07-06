from agents.rag_retriever import retrieve
from agents.granite_agent import (
    extract_symptoms, 
    generate_recommendation,
    detect_language,
    translate_to_english,
    translate_from_english
)
from agents.urgency_classifier import classify_urgency

def run(user_input: str) -> dict:
    # Agent 1: Language detection
    language = detect_language(user_input)
    
    # Agent 2: Translate to English if needed
    english_input = translate_to_english(user_input, language)
    
    # Agent 3: Extract symptoms from english text
    symptoms = extract_symptoms(english_input)
    
    # Agent 4: Retrieve relevant medical context
    retrieved = retrieve(symptoms + " " + english_input, top_k=5)
    context = "\n".join([r['text'] for r in retrieved[:3]])
    possible_diseases = list(set([r['disease'] for r in retrieved if r['disease']]))[:5]
    
    # Agent 5: Classify urgency
    urgency = classify_urgency(symptoms + " " + english_input)
    
    # Agent 6: Generate recommendation
    recommendation_en = generate_recommendation(symptoms, context, urgency['level'])
    
    # Agent 7: Translate recommendation back to user's native language if needed
    if language.lower() != 'english':
        recommendation = translate_from_english(recommendation_en, language)
    else:
        recommendation = recommendation_en
    
    return {
        "language_detected": language,
        "symptoms_detected": symptoms,
        "possible_conditions": possible_diseases,
        "urgency": urgency,
        "recommendation": recommendation,
        "sources_used": len(retrieved),
        "disclaimer": "⚠️ This is educational information only, not a medical diagnosis. Always consult a licensed physician for proper medical advice."
    }

