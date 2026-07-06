import pandas as pd
import json
import os
import xml.etree.ElementTree as ET
import re

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

all_chunks = []

def clean_text(text):
    if not text or str(text).strip() in ['', 'nan']:
        return ''
    return re.sub(r'\s+', ' ', str(text)).strip()

def process_symptom_description_csv(filepath):
    """Handles dataset with Disease, Symptom_1...Symptom_N, Description columns"""
    df = pd.read_csv(filepath)
    chunks = []
    for _, row in df.iterrows():
        disease = clean_text(row.get('Disease', row.get('disease', '')))
        description = clean_text(row.get('Description', row.get('description', '')))
        symptom_cols = [c for c in df.columns 
                       if 'symptom' in c.lower() or 'Symptom' in c]
        symptoms = [clean_text(row[c]) for c in symptom_cols 
                   if clean_text(row[c])]
        if disease:
            text = f"Disease: {disease}."
            if symptoms:
                text += f" Symptoms include: {', '.join(symptoms)}."
            if description:
                text += f" {description[:400]}"
            chunks.append({
                "text": text,
                "disease": disease,
                "symptoms": symptoms,
                "source": "kaggle"
            })
    return chunks

def process_symptom2disease_csv(filepath):
    """Handles dataset with label and text columns (natural language descriptions)"""
    df = pd.read_csv(filepath)
    chunks = []
    for _, row in df.iterrows():
        label = clean_text(row.get('label', row.get('Label', '')))
        text_col = clean_text(row.get('text', row.get('Text', '')))
        if label and text_col:
            text = f"Disease: {label}. Patient description: {text_col[:400]}"
            chunks.append({
                "text": text,
                "disease": label,
                "symptoms": [],
                "source": "symptom2disease"
            })
    return chunks

def process_medlineplus_xml(filepath):
    """Handles MedlinePlus XML files"""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        chunks = []
        for topic in root.findall('health-topic'):
            title = topic.get('title', '')
            summary = topic.findtext('full-summary', '')
            if summary and title:
                clean = re.sub('<.*?>', ' ', summary)
                clean = re.sub(r'\s+', ' ', clean).strip()
                text = f"Health Topic: {title}. {clean[:500]}"
                chunks.append({
                    "text": text,
                    "disease": title,
                    "symptoms": [],
                    "source": "medlineplus"
                })
        return chunks
    except Exception as e:
        print(f"Error processing XML {filepath}: {e}")
        return []

def process_txt_file(filepath):
    """Handles parsing of a single text file containing WHO fact sheets.
    Attempts to identify headings as topics and chunks text paragraphs underneath."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        chunks = []
        current_topic = "WHO Fact Sheet"
        
        # Simple helpers to identify potential header lines:
        def is_header(line):
            line = line.strip()
            if not line:
                return False
            if line.startswith('#'):
                return True
            if re.match(r'^(fact\s*sheet|key\s*facts)', line, re.IGNORECASE):
                return True
            words = line.split()
            if 1 <= len(words) <= 8 and not line.endswith(('.', '?', '!', ';', ',')):
                if any(c.isalpha() for c in line):
                    return True
            return False

        def clean_header(line):
            return re.sub(r'^#+\s*', '', line).strip()

        paragraphs = []
        current_sec = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_sec:
                    paragraphs.append(current_sec)
                    current_sec = []
            else:
                current_sec.append(stripped)
        if current_sec:
            paragraphs.append(current_sec)

        for p in paragraphs:
            if len(p) == 1 and is_header(p[0]):
                current_topic = clean_header(p[0])
                continue
            
            p_text = clean_text(" ".join(p))
            if not p_text:
                continue
            
            words = p_text.split()
            chunk_size = 120  # words per chunk
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                chunk_content = " ".join(chunk_words)
                text = f"Health Topic: {current_topic}. {chunk_content}"
                chunks.append({
                    "text": text,
                    "disease": current_topic,
                    "symptoms": [],
                    "source": "who_fact_sheets"
                })
        return chunks
    except Exception as e:
        print(f"Error processing TXT {filepath}: {e}")
        return []

# Auto-detect and process all files in data/raw/
for filename in os.listdir(RAW_DIR):
    filepath = os.path.join(RAW_DIR, filename)
    if filename.startswith('.'):
        continue
        
    print(f"Processing: {filename}")
    
    if filename.endswith('.csv'):
        df_preview = pd.read_csv(filepath, nrows=1)
        cols = [c.lower() for c in df_preview.columns]
        
        if 'text' in cols and 'label' in cols:
            chunks = process_symptom2disease_csv(filepath)
        else:
            chunks = process_symptom_description_csv(filepath)
        
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks")
    
    elif filename.endswith('.xml'):
        chunks = process_medlineplus_xml(filepath)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks")
        
    elif filename.endswith('.txt'):
        chunks = process_txt_file(filepath)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks")

# Remove duplicates based on text
seen = set()
unique_chunks = []
for chunk in all_chunks:
    if chunk['text'] not in seen:
        seen.add(chunk['text'])
        unique_chunks.append(chunk)

output_path = os.path.join(PROCESSED_DIR, "chunks.json")
with open(output_path, 'w') as f:
    json.dump(unique_chunks, f, indent=2)

print(f"\nDone. Total unique chunks: {len(unique_chunks)}")
print(f"Saved to: {output_path}")
