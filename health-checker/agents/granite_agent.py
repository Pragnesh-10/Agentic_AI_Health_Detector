from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from config import IBM_API_KEY, IBM_PROJECT_ID, IBM_URL

_model = None
_fallback_model = None

def _get_model(model_id="meta-llama/llama-3-3-70b-instruct"):
    credentials = Credentials(url=IBM_URL, api_key=IBM_API_KEY)
    return ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=IBM_PROJECT_ID,
        params={
            "max_new_tokens": 700,
            "temperature": 0.2,
            "repetition_penalty": 1.1
        }
    )

def _generate_text(prompt: str) -> str:
    global _model, _fallback_model
    try:
        if _model is None:
            _model = _get_model("meta-llama/llama-3-3-70b-instruct")
        return _model.generate_text(prompt=prompt)
    except Exception as e:
        print(f"Error with primary model: {e}")
        try:
            if _fallback_model is None:
                _fallback_model = _get_model("meta-llama/llama-3-1-8b")
            return _fallback_model.generate_text(prompt=prompt)
        except Exception as fe:
            print(f"Error with fallback model: {fe}")
            raise fe

def extract_symptoms(user_input: str) -> tuple:
    prompt = f"""<|system|>
You are a medical symptom extractor. Analyze the user's message and:
1. Detect the language they are using.
2. Extract only the medical symptoms.
3. Translate the symptoms to English if they are not already.

Respond strictly in this format:
Language: [Detected Language in English, e.g. English, Spanish, Hindi]
Symptoms: [Comma-separated list of symptoms in English]
<|user|>
Text: "{user_input}"
<|assistant|>"""
    
    result = _generate_text(prompt).strip()
    language = "English"
    symptoms = ""
    
    import re
    lang_match = re.search(r'Language:\s*(.+)', result, re.IGNORECASE)
    if lang_match:
        language = lang_match.group(1).strip()
        
    symp_match = re.search(r'Symptoms:\s*(.+)', result, re.IGNORECASE)
    if symp_match:
        symptoms = symp_match.group(1).strip()
    else:
        # fallback
        symptoms = result
        
    return language, symptoms

def _get_urgency_dict(level: str) -> dict:
    level = level.upper()
    if "EMERGENCY" in level:
        return {
            "level": "EMERGENCY",
            "emoji": "🚨",
            "color": "red",
            "message": "Go to the emergency room immediately or call 112.",
            "action": "Call emergency services now"
        }
    elif "HIGH" in level:
        return {
            "level": "HIGH",
            "emoji": "⚠️",
            "color": "orange",
            "message": "See a doctor today. Do not wait.",
            "action": "Book urgent appointment or visit urgent care"
        }
    else:
        return {
            "level": "MEDIUM",
            "emoji": "🟡",
            "color": "goldenrod",
            "message": "Monitor symptoms. If no improvement in 48 hours, see a doctor.",
            "action": "Rest, hydrate, and monitor"
        }

def generate_recommendation(symptoms: str, context: str, user_language: str = "English") -> tuple:
    prompt = f"""<|system|>
You are a health information assistant providing educational health information. 
You do not diagnose. You provide general health information based on verified sources.
Always recommend consulting a doctor for proper diagnosis.
<|user|>
A user reports these symptoms: {symptoms}

Relevant medical information from verified sources:
{context}

First, assess the urgency of these symptoms.
Urgency Levels:
- EMERGENCY (e.g. chest pain, breathing difficulty, stroke signs)
- HIGH (e.g. high fever, severe pain, confusion)
- MEDIUM (e.g. standard symptoms to monitor)

Provide a structured response. ALWAYS write the final output strictly in the user's language: {user_language}.
Include:
URGENCY: [EMERGENCY, HIGH, or MEDIUM] (Write this part in English to allow system parsing, then write the rest in {user_language})
1. POSSIBLE CONDITIONS: List 2-3 most likely conditions (not a diagnosis)
2. HOME CARE: Simple steps the person can take at home
3. SEE A DOCTOR IF: Specific warning signs that require immediate medical attention
4. GENERAL ADVICE: One practical tip

Keep language simple. No medical jargon.
<|assistant|>"""
    
    result = _generate_text(prompt).strip()
    
    # Extract urgency level
    urgency_level = "MEDIUM"
    import re
    match = re.search(r'URGENCY:\s*(\w+)', result, re.IGNORECASE)
    if match:
        urgency_level = match.group(1).upper()
        
    return result, _get_urgency_dict(urgency_level)



