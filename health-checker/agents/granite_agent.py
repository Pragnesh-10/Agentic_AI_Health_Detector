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

def extract_symptoms(user_input: str) -> str:
    prompt = f"""<|system|>
You are a medical symptom extractor. Extract only the medical symptoms 
from the user's message and return them as a comma-separated list. 
Return only symptoms, nothing else.
<|user|>
Text: "{user_input}"
<|assistant|>
Symptoms:"""
    
    result = _generate_text(prompt).strip()
    if '\n' in result:
        result = result.split('\n')[0]
    return result

def generate_recommendation(symptoms: str, context: str, urgency_level: str) -> str:
    prompt = f"""<|system|>
You are a health information assistant providing educational health information. 
You do not diagnose. You provide general health information based on verified sources.
Always recommend consulting a doctor for proper diagnosis.
<|user|>
A user reports these symptoms: {symptoms}

Relevant medical information from verified sources:
{context}

Urgency level assessed: {urgency_level}

Provide a structured response with:
1. POSSIBLE CONDITIONS: List 2-3 most likely conditions (not a diagnosis)
2. HOME CARE: Simple steps the person can take at home
3. SEE A DOCTOR IF: Specific warning signs that require immediate medical attention
4. GENERAL ADVICE: One practical tip

Keep language simple. No medical jargon.
<|assistant|>"""
    
    return _generate_text(prompt).strip()

def detect_language(text: str) -> str:
    prompt = f"""<|system|>
Detect the language of the text and return only the language name in English. 
Examples: English, Hindi, Telugu, Tamil, Spanish
<|user|>
Text: "{text[:100]}"
<|assistant|>
Language:"""
    result = _generate_text(prompt).strip()
    if '\n' in result:
        result = result.split('\n')[0]
    return result.strip()

def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang.lower() == 'english':
        return text
    prompt = f"""<|system|>
Translate the following text from {source_lang} to English. 
Return only the translation, nothing else.
<|user|>
{text}
<|assistant|>"""
    return _generate_text(prompt).strip()

def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang.lower() == 'english':
        return text
    prompt = f"""<|system|>
Translate the following medical recommendations text from English to {target_lang}. 
Ensure formatting (lists, bullet points, headers) is preserved. 
Return only the translation, nothing else.
<|user|>
{text}
<|assistant|>"""
    return _generate_text(prompt).strip()

