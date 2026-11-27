import re
import csv
import sys
import joblib

sys.stdout.reconfigure(encoding='utf-8')

def chunk_text(title, description, min_words=15, max_words=130, overlap_words=20):
    chunks = []
    
    # Chunk 0: Title
    if title and title.strip():
        chunks.append({
            'type': 'title',
            'section': 'TIÊU ĐỀ',
            'text': f'[TIÊU ĐỀ] {title.strip()}'
        })
    
    if not description or not description.strip():
        return chunks
        
    desc = description.strip()
    
    # 1. Primary splitting by section tags [Header] or Markdown # Header
    # Match [Section Title] at the start of a line
    tag_splits = re.split(r'(?=(?:^|\n)\s*\[[^\]\n]+\])', desc)
    raw_sections = [s.strip() for s in tag_splits if s and s.strip()]
    
    if len(raw_sections) <= 1:
        # Fallback to Markdown headers
        md_splits = re.split(r'(?=(?:^|\n)\s*#{1,6}\s+)', desc)
        raw_sections = [s.strip() for s in md_splits if s and s.strip()]
        
    if len(raw_sections) <= 1:
        # Fallback to double newlines (paragraphs)
        p_splits = re.split(r'\n\s*\n+', desc)
        raw_sections = [s.strip() for s in p_splits if s and s.strip()]

    # 2. Process sections and sub-chunk if needed
    accumulated_short = ""
    accumulated_header = ""
    
    for raw_sec in raw_sections:
        lines = [l.strip() for l in raw_sec.split('\n') if l.strip()]
        if not lines:
            continue
            
        first_line = lines[0]
        header = ""
        body_lines = lines
        if first_line.startswith('[') and first_line.endswith(']'):
            header = first_line
            body_lines = lines[1:]
        elif first_line.startswith('#'):
            header = first_line.lstrip('#').strip()
            body_lines = lines[1:]
            
        sec_body = '\n'.join(body_lines).strip()
        sec_words = (header + " " + sec_body).split()
        
        # If very short (e.g. just a lone header or <= 10 words), accumulate it
        if len(sec_words) < min_words:
            if accumulated_short:
                accumulated_short += "\n\n" + raw_sec
            else:
                accumulated_short = raw_sec
                accumulated_header = header or "Nội dung"
            continue
            
        # If we had accumulated short text, prepend or flush it
        if accumulated_short:
            if len(accumulated_short.split()) + len(sec_words) <= max_words:
                raw_sec = accumulated_short + "\n\n" + raw_sec
                header = accumulated_header or header
                words = raw_sec.split()
                accumulated_short = ""
                accumulated_header = ""
            else:
                chunks.append({
                    'type': 'section',
                    'section': accumulated_header or 'Nội dung',
                    'text': accumulated_short
                })
                accumulated_short = ""
                accumulated_header = ""
                
        words = raw_sec.split()
        
        # If section fits in max_words, keep as single chunk
        if len(words) <= max_words:
            chunks.append({
                'type': 'section',
                'section': header or 'Nội dung',
                'text': raw_sec
            })
        else:
            # Overly long section: split into sliding windows by sentences or lines
            cur_tokens = []
            cur_lines = []
            
            for line in lines:
                line_tokens = line.split()
                if len(cur_tokens) + len(line_tokens) > max_words and cur_tokens:
                    chunk_text_body = '\n'.join(cur_lines)
                    full_chunk = f"{header}\n{chunk_text_body}" if header and not chunk_text_body.startswith(header) else chunk_text_body
                    chunks.append({
                        'type': 'sub_section',
                        'section': header or 'Nội dung dài',
                        'text': full_chunk
                    })
                    overlap_tokens = cur_tokens[-overlap_words:]
                    cur_tokens = list(overlap_tokens) + line_tokens
                    cur_lines = [line]
                else:
                    cur_tokens.extend(line_tokens)
                    cur_lines.append(line)
                    
            if cur_lines:
                chunk_text_body = '\n'.join(cur_lines)
                full_chunk = f"{header}\n{chunk_text_body}" if header and not chunk_text_body.startswith(header) else chunk_text_body
                chunks.append({
                    'type': 'sub_section',
                    'section': header or 'Nội dung dài',
                    'text': full_chunk
                })
                
    if accumulated_short:
        chunks.append({
            'type': 'section',
            'section': accumulated_header or 'Nội dung',
            'text': accumulated_short
        })
        
    return chunks

# Test on Job 13
title = 'Chuyên Viên Phát Triển Khách Hàng Cá Nhân - MRBT'
desc = ''
with open('job13_details.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        t = r['title']
        c = r['content']
        desc += f'[{t}]\n{c}\n\n'

chunks = chunk_text(title, desc)
print(f'Total chunks created: {len(chunks)}')

vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
model = joblib.load('models/multilabel_model.joblib')
LABEL_COLUMNS = ['UPFRONT_FEE_REQUEST', 'JOB_SCAM', 'PYRAMID_INVEST', 'TOXIC_CONTENT', 'DISCRIMINATION', 'SPAM_ADS']

for i, c in enumerate(chunks):
    words = len(c['text'].split())
    X = vectorizer.transform([c['text']])
    probs = model.predict_proba(X)[0]
    preds = model.predict(X)[0]
    print(f"\nChunk {i} [{c['section']}] ({c['type']}, words={words}):")
    print("  Preview:", c['text'][:70].replace('\n', ' '))
    for j, l in enumerate(LABEL_COLUMNS):
        if probs[j] >= 0.3:
            print(f"    {l}: pred={preds[j]}, prob={probs[j]:.4f}")
