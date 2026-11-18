import re
import numpy as np

LABEL_COLUMNS = [
    "UPFRONT_FEE_REQUEST",
    "JOB_SCAM",
    "PYRAMID_INVEST",
    "TOXIC_CONTENT",
    "DISCRIMINATION",
    "SPAM_ADS"
]

def chunk_text(title: str, description: str, min_words: int = 15, max_words: int = 130, overlap_words: int = 20):
    """
    Phân đoạn văn bản bài tuyển dụng theo ranh giới ngữ nghĩa:
    1. Chunk 0: Tiêu đề bài viết.
    2. Primary Chunking: Tách theo tiêu đề mục [Tên mục], Markdown (#, ##), hoặc đoạn văn (\n\n).
    3. Sub-chunking: Với các mục quá dài (> max_words), tách tiếp theo dòng/câu với sliding window có overlap.
    4. Tránh phân mảnh: Các đoạn quá ngắn (< min_words) được gộp với đoạn liền kề.
    """
    chunks = []
    
    # 1. Chunk 0: Tiêu đề bài viết
    if title and title.strip():
        chunks.append({
            'type': 'title',
            'section': 'TIÊU ĐỀ',
            'text': f'[TIÊU ĐỀ] {title.strip()}'
        })
    
    if not description or not description.strip():
        return chunks
        
    desc = description.strip()
    
    # 2. Tách theo thẻ mục [Tên mục] hoặc Markdown headers
    tag_splits = re.split(r'(?=(?:^|\n)\s*\[[^\]\n]+\])', desc)
    raw_sections = [s.strip() for s in tag_splits if s and s.strip()]
    
    if len(raw_sections) <= 1:
        md_splits = re.split(r'(?=(?:^|\n)\s*#{1,6}\s+)', desc)
        raw_sections = [s.strip() for s in md_splits if s and s.strip()]
        
    if len(raw_sections) <= 1:
        p_splits = re.split(r'\n\s*\n+', desc)
        raw_sections = [s.strip() for s in p_splits if s and s.strip()]

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
        
        # Xử lý đoạn quá ngắn
        if len(sec_words) < min_words:
            if accumulated_short:
                accumulated_short += "\n\n" + raw_sec
            else:
                accumulated_short = raw_sec
                accumulated_header = header or "Nội dung"
            continue
            
        if accumulated_short:
            if len(accumulated_short.split()) + len(sec_words) <= max_words:
                raw_sec = accumulated_short + "\n\n" + raw_sec
                header = accumulated_header or header
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
        
        if len(words) <= max_words:
            chunks.append({
                'type': 'section',
                'section': header or 'Nội dung',
                'text': raw_sec
            })
        else:
            # Phân tách đoạn dài thành sliding window
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

def is_explicit_fee_negation(text: str) -> bool:
    """
    Bộ lọc chống False Positive: Kiểm tra các tuyên bố minh bạch không thu phí.
    Nếu đoạn văn chứa khẳng định miễn phí/không thu phí và KHÔNG chứa hành vi đòi tiền,
    tránh để bag-of-words bị kích hoạt nhầm bởi các từ 'phí', 'khoản', 'tài khoản'.
    """
    lower = text.lower()
    
    negation_patterns = [
        "không thu bất kỳ khoản phí",
        "không thu phí",
        "không mất phí",
        "hoàn toàn miễn phí",
        "không yêu cầu đóng phí",
        "không yêu cầu nộp phí",
        "không yêu cầu thanh toán",
        "không yêu cầu chuyển khoản",
        "không yêu cầu nộp tiền",
        "không yêu cầu đặt cọc",
        "tuyệt đối không thu phí",
        "tuyệt đối không yêu cầu",
        "không thu tiền",
        "không phải ứng vốn",
        "không ứng vốn",
        "không cọc",
        "không phải cọc",
        "không cần cọc",
        "miễn phí 100%"
    ]
    has_negation = any(p in lower for p in negation_patterns)
    
    affirmative_patterns = [
        "cần chuyển khoản",
        "phải chuyển khoản",
        "chuyển khoản trước",
        "nộp trước",
        "nộp phí",
        "đóng phí",
        "đóng tiền",
        "đặt cọc",
        "nạp tiền để",
        "phí kích hoạt",
        "phí hồ sơ",
        "phí đồng phục",
        "phí giữ chỗ",
        "tiền thế chân"
    ]
    has_affirmative = any(p in lower for p in affirmative_patterns)
    
    return has_negation and not has_affirmative

def moderate_job_post(title: str, description: str, vectorizer, model, threshold: float = 0.5):
    """
    Thực hiện kiểm duyệt bài tuyển dụng theo kiến trúc Chunk-based Classification:
    1. Chia văn bản thành các đoạn ngữ nghĩa (semantic chunks).
    2. Dự đoán xác suất 6 nhãn trên từng đoạn độc lập.
    3. Áp dụng bộ lọc chống False Positive theo ngữ cảnh.
    4. Tổng hợp xác suất cực đại (Max-Pooling) kết hợp đánh giá toàn văn bản.
    5. Trả về kết quả hoàn toàn tương thích với API contract hiện có của Spring Boot.
    """
    chunks = chunk_text(title, description)
    
    # Trường hợp không thể phân đoạn
    if not chunks:
        full_text = f"[TIÊU ĐỀ] {title} [MÔ TẢ] {description}"
        X = vectorizer.transform([full_text])
        preds = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        return {
            "title": title,
            "results": {
                LABEL_COLUMNS[i]: {
                    "prediction": int(preds[i]),
                    "probability": round(float(probs[i]), 4)
                } for i in range(len(LABEL_COLUMNS))
            }
        }
        
    chunk_probs = []
    
    for c in chunks:
        X = vectorizer.transform([c['text']])
        p = model.predict_proba(X)[0].copy()
        
        # Bảo vệ false positive cho UPFRONT_FEE_REQUEST nếu đoạn văn là cam kết không thu phí
        if is_explicit_fee_negation(c['text']):
            p[0] = min(p[0], 0.25)
            
        chunk_probs.append(p)
        
    chunk_probs = np.array(chunk_probs) # shape: (num_chunks, 6)
    
    # Đánh giá thêm toàn văn bản để không mất tín hiệu toàn cục nếu bài đăng ngắn
    full_text = f"[TIÊU ĐỀ] {title} [MÔ TẢ] {description}"
    p_whole = model.predict_proba(vectorizer.transform([full_text]))[0]
    
    # Tổng hợp xác suất cực đại (Max Probability Aggregation)
    max_chunk_probs = chunk_probs.max(axis=0)
    final_probs = np.maximum(max_chunk_probs, p_whole)
    
    results = {}
    for i, label in enumerate(LABEL_COLUMNS):
        prob = float(final_probs[i])
        pred = 1 if prob >= threshold else 0
        results[label] = {
            "prediction": pred,
            "probability": round(prob, 4)
        }
        
    return {
        "title": title,
        "results": results
    }

