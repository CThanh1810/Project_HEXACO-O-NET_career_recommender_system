import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import định nghĩa 24 facets và 21 workstyles
from mapping_hexaco_to_workstyles import hexaco_facets, onet_workstyles

# 1. Định vị đường dẫn file an toàn
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

output_path = project_root / "data" / "gemini_mapping_weights.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# 2. Khởi tạo Gemini Client
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Chưa tìm thấy GEMINI_API_KEY! Hãy cấu hình trong file .env")

client = genai.Client(api_key=api_key)

# Bạn có thể chọn gemini-3.8-flash hoặc gemini-4-argon
MODEL_NAME = "gemini-3.8-flash"

facet_names = list(hexaco_facets.keys())
hexaco_context = "\n".join([f"- {name}: {desc}" for name, desc in hexaco_facets.items()])

# 3. Đọc tiến trình cũ nếu file đã tồn tại
saved_data = {
    "model_used": MODEL_NAME,
    "facets_order": facet_names,
    "workstyles": list(onet_workstyles.keys()),
    "details": {}
}

if output_path.exists():
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            if "details" in existing:
                saved_data["details"] = existing["details"]
                print(f"📦 Đã tìm thấy file tiến trình cũ: Đã có {len(saved_data['details'])}/21 phong cách làm việc.")
    except Exception as e:
        print(f"⚠️ Không đọc được file cũ, sẽ tạo mới: {e}")

def save_checkpoint():
    """Hàm lưu ngay lập tức xuống đĩa sau mỗi lần gọi thành công"""
    # Xây dựng lại matrix theo đúng thứ tự 21 workstyles
    matrix_rows = []
    for ws in onet_workstyles.keys():
        if ws in saved_data["details"]:
            matrix_rows.append(saved_data["details"][ws]["weights"])
            
    payload = {
        "model_used": MODEL_NAME,
        "facets_order": facet_names,
        "workstyles": list(onet_workstyles.keys()),
        "matrix": matrix_rows,
        "details": saved_data["details"],
        "completed": f"{len(saved_data['details'])}/21"
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


print(f"\n🚀 Bắt đầu tiến trình đánh giá với model: {MODEL_NAME}...\n")

for idx, (ws_name, ws_desc) in enumerate(onet_workstyles.items(), 1):
    # NẾU ĐÃ CÓ TRONG FILE -> BỎ QUA LUÔN, KHÔNG GỌI API
    if ws_name in saved_data["details"] and len(saved_data["details"][ws_name].get("weights", [])) == 24:
        print(f"[{idx}/21] ⏭️ Đã có sẵn trong checkpoint, bỏ qua: {ws_name}")
        continue
    
    print(f"[{idx}/21] ⏳ Đang gọi API đánh giá: {ws_name}...")
    
    prompt = f"""You are an expert Industrial-Organizational (I/O) Psychologist and Psychometrics Researcher.

Task:
Evaluate the directional relationship between 24 HEXACO personality facets and the specific O*NET Work Style: "{ws_name}".

O*NET Work Style Definition:
"{ws_desc}"

24 HEXACO Facets Definitions:
{hexaco_context}

Scoring Rubric (weights between -1.0 and +1.0, step of 0.1):
- [+0.8 to +1.0]: Strong positive predictor (high facet strongly drives high performance in this work style).
- [+0.3 to +0.7]: Moderate positive predictor.
- [ 0.0 ]: Unrelated or negligible impact.
- [-0.3 to -0.7]: Moderate negative predictor.
- [-0.8 to -1.0]: Strong negative predictor (high facet strongly hinders this work style).

Facets to evaluate in exact order (24 items):
{facet_names}

Output Requirement:
Return strictly a valid JSON object matching this schema:
{{
  "weights": [24 float numbers between -1.0 and 1.0 corresponding to the 24 facets],
  "reasoning": "Brief explanation of the top 2-3 most influential facets for this work style."
}}
"""
    # Cơ chế tự động thử lại (Retry) tối đa 5 lần nếu API bị High Demand / Quá tải
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            
            data = json.loads(response.text)
            
            # Cập nhật kết quả vào bộ nhớ
            saved_data["details"][ws_name] = {
                "weights": data["weights"],
                "reasoning": data.get("reasoning", "")
            }
            
            # GHI NGAY LẬP TỨC VÀO ĐĨA!
            save_checkpoint()
            print(f"[{idx}/21] ✅ Đã xong & LƯU FILE THÀNH CÔNG: {ws_name}")
            
            # Nghỉ 2 giây giữa các request để giảm tải cho server
            time.sleep(2)
            break
            
        except Exception as e:
            if attempt < max_retries:
                wait_sec = attempt * 6  # Chờ 6s, 12s, 18s, 24s...
                print(f"⚠️ Máy chủ Google bị nghẽn/High Demand ({e}). Đang tự thử lại lần {attempt}/{max_retries} sau {wait_sec} giây...")
                time.sleep(wait_sec)
            else:
                print(f"❌ Đã thử lại {max_retries} lần không thành công tại tiêu chí: {ws_name}. Dừng lại để bảo toàn dữ liệu!")
                save_checkpoint()
                raise e

print(f"\n🎉 HOÀN THÀNH TẤT CẢ! Ma trận đã được lưu trọn vẹn tại: {output_path}")