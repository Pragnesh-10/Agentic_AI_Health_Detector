import pandas as pd
import json
import xml.etree.ElementTree as ET

def process_kaggle_csv(filepath):
    df = pd.read_csv(filepath)
    chunks = []
    
    for _, row in df.iterrows():
        disease = row.get('Disease', row.get('disease', ''))
        description = row.get('Description', row.get('description', ''))
        
        # Collect all symptom columns
        symptom_cols = [c for c in df.columns if 'Symptom' in c or 'symptom' in c]
        symptoms = [str(row[c]).strip() for c in symptom_cols 
                   if pd.notna(row[c]) and str(row[c]).strip() not in ['', 'nan']]
        
        chunk = f"Disease: {disease}. Symptoms: {', '.join(symptoms)}. {description}"
        chunks.append({"text": chunk, "disease": disease, "symptoms": symptoms})
    
    return chunks

def process_medlineplus_xml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    chunks = []
    
    for topic in root.findall('health-topic'):
        title = topic.get('title', '')
        summary = topic.findtext('full-summary', '')
        if summary and title:
            # Clean HTML tags from summary
            import re
            clean = re.sub('<.*?>', '', summary)
            chunk = f"Health Topic: {title}. {clean[:500]}"  # limit chunk size
            chunks.append({"text": chunk, "disease": title})
    
    return chunks

# Run it
kaggle_chunks = process_kaggle_csv('data/raw/disease_symptom_description.csv')
medline_chunks = process_medlineplus_xml('data/raw/mplus_topics.xml')

all_chunks = kaggle_chunks + medline_chunks

with open('data/processed/chunks.json', 'w') as f:
    json.dump(all_chunks, f)

print(f"Total chunks created: {len(all_chunks)}")