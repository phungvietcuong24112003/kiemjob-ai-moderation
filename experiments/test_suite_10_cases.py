import re
import sys
import joblib
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Load models
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
model = joblib.load('models/multilabel_model.joblib')

LABEL_COLUMNS = [
    "UPFRONT_FEE_REQUEST",
    "JOB_SCAM",
    "PYRAMID_INVEST",
    "TOXIC_CONTENT",
    "DISCRIMINATION",
    "SPAM_ADS"
]

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
    
    # Primary splitting by section tags [Header] or Markdown # Header
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

# Negation & false positive guard for fee requests
# Checks if the text strictly negates fee request or discusses company salary payout
def is_explicit_fee_negation(text):
    lower = text.lower()
    # Explicit negation phrases where company asserts no fees
    negation_patterns = [
        "không thu bất kỳ khoản phí",
        "không thu phí",
        "không mất phí",
        "hoàn toàn miễn phí",
        "không yêu cầu đóng phí",
        "không yêu cầu chuyển khoản trước",
        "không cọc",
        "không thu tiền",
        "miễn phí 100%"
    ]
    has_negation = any(p in lower for p in negation_patterns)
    
    # Check if there is also an affirmative fee request phrase
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
    
    # If it has negation but NO affirmative fee request, it's a disclaimer!
    if has_negation and not has_affirmative:
        return True
    return False

def predict_whole_doc(title, description):
    text = f"[TIÊU ĐỀ] {title} [MÔ TẢ] {description}"
    X = vectorizer.transform([text])
    preds = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    return {
        LABEL_COLUMNS[i]: {
            'prediction': int(preds[i]),
            'probability': round(float(probs[i]), 4)
        } for i in range(len(LABEL_COLUMNS))
    }

def predict_chunk_based(title, description, threshold=0.5):
    chunks = chunk_text(title, description)
    if not chunks:
        return predict_whole_doc(title, description)
        
    chunk_results = []
    chunk_probs = []
    
    for c in chunks:
        X = vectorizer.transform([c['text']])
        p = model.predict_proba(X)[0]
        
        # Guard: check if chunk is merely a transparency disclaimer ("không thu phí")
        if is_explicit_fee_negation(c['text']):
            p[0] = min(p[0], 0.25) # UPFRONT_FEE_REQUEST dampened
            
        chunk_probs.append(p)
        chunk_results.append({
            'section': c['section'],
            'type': c['type'],
            'text': c['text'],
            'probabilities': p
        })
        
    chunk_probs = np.array(chunk_probs) # shape: (num_chunks, 6)
    
    # Also evaluate whole document as one candidate
    text_whole = f"[TIÊU ĐỀ] {title} [MÔ TẢ] {description}"
    p_whole = model.predict_proba(vectorizer.transform([text_whole]))[0]
    
    # Aggregate: Max probability between max(chunk) and whole doc
    max_chunk_probs = chunk_probs.max(axis=0)
    final_probs = np.maximum(max_chunk_probs, p_whole)
    
    final_results = {}
    triggered_chunks = {}
    
    for i, label in enumerate(LABEL_COLUMNS):
        prob = float(final_probs[i])
        pred = 1 if prob >= threshold else 0
        final_results[label] = {
            'prediction': pred,
            'probability': round(prob, 4)
        }
        # Find which chunk gave the max score
        best_chunk_idx = int(chunk_probs[:, i].argmax())
        triggered_chunks[label] = {
            'section': chunks[best_chunk_idx]['section'],
            'chunk_prob': round(float(chunk_probs[best_chunk_idx, i]), 4),
            'sample_text': chunks[best_chunk_idx]['text'][:80].replace('\n', ' ')
        }
        
    return {
        'results': final_results,
        'triggered_chunks': triggered_chunks,
        'num_chunks': len(chunks)
    }

# =========================================================================
# 10 TEST CASES (THEO YÊU CẦU MỤC 7)
# =========================================================================

scam_fee_snippet = """Công ty đang tuyển nhân viên nhập liệu làm việc tại nhà.
Không yêu cầu kinh nghiệm, được đào tạo từ đầu.
Thu nhập từ 10 đến 15 triệu đồng mỗi tháng.
Ứng viên cần chuyển khoản trước 500.000 đồng phí hồ sơ và phí kích hoạt tài khoản trước khi được nhận việc.
Sau khi thanh toán, công ty sẽ gửi tài khoản làm việc và hướng dẫn công việc."""

msb_section_1 = """[Mô tả công việc]
1. Chăm sóc, phục vụ khách hàng trong danh mục quản lý nhằm đảm bảo các quy định, tiêu chuẩn chất lượng dịch vụ của MSB.
2. Lập kế hoạch kinh doanh, phát triển khách hàng, tiếp xúc, giới thiệu, tư vấn, bán các sản phẩm của MSB, trọng tâm là sản phẩm tín dụng tín chấp, thế chấp và các sản phẩm bán chéo khác nhằm hoàn thành chỉ tiêu được giao.
3. Đảm bảo việc tư vấn, hướng dẫn, chọn lọc khách hàng chất lượng để nhiều khách hàng được phê duyệt cho vay tín dụng và sử dụng nhiều sản phẩm của ngân hàng.
4. Xây dựng, mở rộng tập khách hàng mới, hiện hữu thông qua tư vấn gói giải pháp tiện ích toàn diện.
5. Duy trì và phát triển các mối quan hệ từ khách hàng hiện hữu và khách hàng tiềm năng."""

msb_section_2 = """[Yêu cầu ứng viên]
1. Trình độ: Cao đẳng trở lên cho các chuyên ngành: Tài chính - Ngân hàng, Quản trị Kinh doanh, Các khối ngành Kinh tế nói chung.
2. Kinh nghiệm: Có kinh nghiệm bán các sản phẩm tín dụng hoặc bán hàng trong các lĩnh vực Tài chính, Ngân hàng, Bảo hiểm.
3. Năng lực: Có khả năng đàm phán, thuyết phục tốt, có khả năng đọc và phân tích báo cáo tài chính, có khả năng nhận biết và quản trị rủi ro."""

msb_section_3 = """[Thu nhập]
Thu nhập khi đạt 100% KPI: Thoả thuận theo năng lực.
Thu nhập tính theo tỷ lệ đạt KPI và hiệu quả kinh doanh.
Lương cứng không phụ thuộc doanh số bán hàng."""

msb_section_4 = """[Quyền lợi được hưởng]
1. Thu nhập cạnh tranh – Đãi ngộ xứng tầm: Thỏa thuận mức lương hấp dẫn, tương xứng với năng lực và đóng góp thực tế.
2. Cơ hội thăng tiến rộng mở, lộ trình nghề nghiệp rõ ràng và minh bạch theo định kỳ.
3. Chế độ đãi ngộ toàn diện: Gói bảo hiểm sức khỏe cao cấp dành riêng cho CBNV và người thân.
4. Khám sức khỏe định kỳ hàng năm tại các bệnh viện uy tín chất lượng cao.
5. Môi trường làm việc nhân văn, chuyên nghiệp, gắn kết và tự hào."""

test_cases = [
    {
        "id": 1,
        "name": "1. Bài đăng ngắn có yêu cầu đóng phí trước",
        "title": "Tuyển CTV nhập liệu tại nhà",
        "description": scam_fee_snippet,
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 1
    },
    {
        "id": 2,
        "name": "2. Bài đăng dài có yêu cầu đóng phí ở CUỐI",
        "title": "Chuyên Viên Phát Triển Khách Hàng Cá Nhân - MRBT",
        "description": f"{msb_section_1}\n\n{msb_section_2}\n\n{msb_section_3}\n\n{msb_section_4}\n\n[Thông tin bổ sung]\n{scam_fee_snippet}",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 1
    },
    {
        "id": 3,
        "name": "3. Bài đăng dài có yêu cầu đóng phí ở GIỮA",
        "title": "Chuyên Viên Phát Triển Khách Hàng Cá Nhân - MRBT",
        "description": f"{msb_section_1}\n\n{msb_section_2}\n\n[Lưu ý quan trọng]\n{scam_fee_snippet}\n\n{msb_section_3}\n\n{msb_section_4}",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 1
    },
    {
        "id": 4,
        "name": "4. Bài đăng dài có yêu cầu đóng phí ở ĐẦU",
        "title": "Chuyên Viên Phát Triển Khách Hàng Cá Nhân - MRBT",
        "description": f"[Thông báo tuyển dụng]\n{scam_fee_snippet}\n\n{msb_section_1}\n\n{msb_section_2}\n\n{msb_section_3}\n\n{msb_section_4}",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 1
    },
    {
        "id": 5,
        "name": "5. Bài đăng dài hợp lệ (MSB thuần túy, không vi phạm)",
        "title": "Chuyên Viên Phát Triển Khách Hàng Cá Nhân - MRBT",
        "description": f"{msb_section_1}\n\n{msb_section_2}\n\n{msb_section_3}\n\n{msb_section_4}",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 0
    },
    {
        "id": 6,
        "name": "6. Bài đăng hợp lệ có mức lương cao (Senior IT)",
        "title": "Senior Solutions Architect / Tech Lead lương $3000 - $5000",
        "description": """[Mô tả công việc]
Chịu trách nhiệm thiết kế kiến trúc hệ thống phân tán microservices quy mô lớn, chịu tải hàng triệu request/ngày.
Lãnh đạo đội ngũ kỹ sư 15 thành viên, tối ưu hạ tầng cloud AWS và Kubernetes.
[Yêu cầu ứng viên]
Tối thiểu 7 năm kinh nghiệm phát triển phần mềm Java / Golang.
Có kinh nghiệm sâu sắc về Kafka, Redis, PostgreSQL và kiến trúc Cloud-native.
[Quyền lợi]
Mức lương từ 70.000.000 đến 120.000.000 VNĐ/tháng kèm cổ phiếu ESOP thưởng hàng năm.
Làm việc hybrid linh hoạt, hỗ trợ thiết bị Macbook Pro M3 Max và ngân sách học tập $1000/năm.""",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 0
    },
    {
        "id": 7,
        "name": "7. Bài đăng tuyển dụng ngân hàng hợp lệ (Vietcombank)",
        "title": "Giao Dịch Viên Ngân Hàng - Hội Sở Chính",
        "description": """[Mô tả công việc]
Tiếp đón khách hàng tại quầy giao dịch, thực hiện các giao dịch nộp rút tiền mặt, chuyển khoản ngân hàng, mở sổ tiết kiệm.
Tư vấn phát hành thẻ tín dụng, ngân hàng điện tử VCB Digibank cho khách hàng cá nhân.
[Yêu cầu ứng viên]
Tốt nghiệp Đại học chính quy chuyên ngành Tài chính, Ngân hàng, Kế toán.
Ngoại hình sáng, giao tiếp tự tin, chuẩn mực tác phong ngân hàng.
[Quyền lợi]
Lương và thưởng cạnh tranh theo kết quả kinh doanh định kỳ quý và năm.
Hưởng đầy đủ chế độ bảo hiểm cao cấp, du lịch nghỉ dưỡng hàng năm cùng công đoàn.""",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 0
    },
    {
        "id": 8,
        "name": "8. Nhắc từ 'chuyển khoản' nhưng là công ty trả lương",
        "title": "Nhân Viên Thu Ngân Cửa Hàng Bách Hóa",
        "description": """[Mô tả công việc]
Thực hiện thanh toán tiền hàng cho khách hàng bằng tiền mặt, quẹt thẻ POS hoặc quét mã QR chuyển khoản.
Kiểm kê hàng hóa đầu ca và cuối ca làm việc.
[Thu nhập & Thanh toán]
Lương cứng 7.500.000 VNĐ + phụ cấp cơm trưa 30.000 VNĐ/ngày.
Lương được chuyển khoản trực tiếp qua tài khoản ngân hàng của nhân viên vào ngày mùng 5 hàng tháng.
Công ty cam kết không thu bất kỳ khoản phí nào của người lao động khi nộp hồ sơ phỏng vấn.""",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 0
    },
    {
        "id": 9,
        "name": "9. Đào tạo, cấp tài khoản làm việc nhưng KHÔNG thu phí",
        "title": "Chuyên Viên Chăm Sóc Khách Hàng E-Commerce",
        "description": """[Mô tả công việc]
Trực tổng đài và trả lời tin nhắn giải đáp thắc mắc của người tiêu dùng trên sàn thương mại điện tử.
Làm việc theo ca xoay tại văn phòng công ty.
[Đào tạo & Hỗ trợ]
Ứng viên trúng tuyển được tham gia khóa đào tạo nghiệp vụ 5 ngày có hưởng nguyên lương.
Công ty cấp tài khoản làm việc, máy tính để bàn và tai nghe chuyên dụng miễn phí.
Tuyệt đối không yêu cầu ứng viên thanh toán hay nộp tiền dưới bất kỳ hình thức nào.""",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 0
    },
    {
        "id": 10,
        "name": "10. Yêu cầu thanh toán trước bằng cách diễn đạt khác (đồng phục, thế chân)",
        "title": "Tuyển Nhân Viên Phục Vụ Nhà Hàng Tiệc Cưới",
        "description": """[Mô tả công việc]
Set up bàn tiệc, bưng bê món ăn và phục vụ khách trong các tiệc cưới cuối tuần.
Thời gian làm việc theo ca linh hoạt (ca 4 tiếng hoặc ca 8 tiếng).
Lương thanh toán sau ca: 250.000 - 400.000 VNĐ/tiệc.
[Yêu cầu nhận việc]
Để được cấp phát đồng phục nhà hàng và giữ lịch làm việc, nhân viên cần đóng trước 300.000 đồng tiền thế chân đồng phục và làm thẻ nhân viên khi đến văn phòng nhận việc.""",
        "expected_label": "UPFRONT_FEE_REQUEST",
        "expected_pred": 1
    }
]

print("=" * 80)
print("KIỂM THỬ ĐỐI CHỨNG 10 TRƯỜNG HỢP: CŨ (WHOLE-DOC) VS MỚI (CHUNK-BASED)")
print("=" * 80)

correct_old = 0
correct_new = 0

for tc in test_cases:
    # 1. Old method
    res_old = predict_whole_doc(tc['title'], tc['description'])
    pred_old = res_old[tc['expected_label']]['prediction']
    prob_old = res_old[tc['expected_label']]['probability']
    is_ok_old = (pred_old == tc['expected_pred'])
    if is_ok_old: correct_old += 1
    
    # 2. New Chunk-based method
    res_new = predict_chunk_based(tc['title'], tc['description'])
    pred_new = res_new['results'][tc['expected_label']]['prediction']
    prob_new = res_new['results'][tc['expected_label']]['probability']
    trig_sec = res_new['triggered_chunks'][tc['expected_label']]['section']
    is_ok_new = (pred_new == tc['expected_pred'])
    if is_ok_new: correct_new += 1
    
    status_old = "PASSED" if is_ok_old else "FAILED (X)"
    status_new = "PASSED" if is_ok_new else "FAILED (X)"
    
    print(f"\nTest Case {tc['id']}: {tc['name']}")
    print(f"  Target Label: {tc['expected_label']} (Expected: {tc['expected_pred']})")
    print(f"  [CŨ - WholeDoc]    : Pred={pred_old}, Prob={prob_old:.4f} -> {status_old}")
    print(f"  [MỚI - ChunkBased] : Pred={pred_new}, Prob={prob_new:.4f} (Trigger: {trig_sec}, Chunks: {res_new['num_chunks']}) -> {status_new}")

print("\n" + "=" * 80)
print(f"KẾT QUẢ TỔNG KẾT:")
print(f"  Phương pháp CŨ (Whole-Doc)   : {correct_old} / {len(test_cases)} đúng ({(correct_old/len(test_cases))*100:.1f}%)")
print(f"  Phương pháp MỚI (Chunk-Based): {correct_new} / {len(test_cases)} đúng ({(correct_new/len(test_cases))*100:.1f}%)")
print("=" * 80)

