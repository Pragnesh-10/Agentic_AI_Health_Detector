EMERGENCY_PATTERNS = [
    "chest pain", "cannot breathe", "can't breathe", "difficulty breathing",
    "shortness of breath", "unconscious", "not responding", "severe allergic",
    "anaphylaxis", "stroke", "face drooping", "arm weakness", "slurred speech",
    "paralysis", "heart attack", "seizure", "fitting", "coughing blood",
    "vomiting blood", "blood in stool", "loss of consciousness", "overdose",
    "suspected poisoning"
]

HIGH_PATTERNS = [
    "fever above 103", "fever above 39", "high fever for 3 days",
    "fever in infant", "fever in baby", "blood in urine", "severe headache",
    "sudden vision loss", "confusion", "disoriented", "stiff neck with fever",
    "rash spreading", "yellowing skin", "jaundice", "severe abdominal pain",
    "pain that wont go away"
]

def classify_urgency(text: str) -> dict:
    text_lower = text.lower()
    
    for pattern in EMERGENCY_PATTERNS:
        if pattern in text_lower:
            return {
                "level": "EMERGENCY",
                "emoji": "🚨",
                "color": "red",
                "message": "Go to the emergency room immediately or call 112.",
                "action": "Call emergency services now"
            }
    
    for pattern in HIGH_PATTERNS:
        if pattern in text_lower:
            return {
                "level": "HIGH",
                "emoji": "⚠️",
                "color": "orange",
                "message": "See a doctor today. Do not wait.",
                "action": "Book urgent appointment or visit urgent care"
            }
    
    return {
        "level": "MEDIUM",
        "emoji": "🟡",
        "color": "goldenrod",
        "message": "Monitor symptoms. If no improvement in 48 hours, see a doctor.",
        "action": "Rest, hydrate, and monitor"
    }
