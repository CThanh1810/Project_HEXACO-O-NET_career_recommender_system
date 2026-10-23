import json
from pathlib import Path
import numpy as np
from scipy.stats import norm

# 1. Đường dẫn an toàn tới file ma trận vừa sinh
project_root = Path(__file__).resolve().parent.parent
weights_path = project_root / "data" / "gemini_mapping_weights.json"

with open(weights_path, "r", encoding="utf-8") as f:
    gemini_data = json.load(f)

# Ma trận 21 x 24 do Gemini sinh ra
WEIGHT_MATRIX = np.array(gemini_data["matrix"])
# Tên 21 workstyles chuẩn chữ thường snake_case
WORKSTYLE_NAMES = [ws.lower().replace(" ", "_").replace("-","_") for ws in gemini_data["workstyles"]]

def predict_workstyles(user_hexaco_scores):
    """
    Quy đổi 24 điểm chuẩn hóa HEXACO (Mean=5.0, SD=1.0) 
    sang 21 điểm O*NET Work Styles (thang 0-100) qua phân vị Norm CDF.
    """
    user_z = np.array(user_hexaco_scores) - 5.0
    raw_dot = np.dot(WEIGHT_MATRIX, user_z)
    
    sum_abs_weights = np.sum(np.abs(WEIGHT_MATRIX), axis=1)
    sum_abs_weights = np.where(sum_abs_weights == 0, 1.0, sum_abs_weights)
    
    workstyle_z = raw_dot / sum_abs_weights
    # Đưa về thang điểm O*NET qua hàm tích lũy xác suất chuẩn (Percentile 0-100)
    predicted_onet_scores = norm.cdf(workstyle_z) * 100
    
    return dict(zip(WORKSTYLE_NAMES, np.round(predicted_onet_scores, 2)))

# Bộ điểm thực tế từ bài test trên ảnh của bạn:
MY_HEXACO_SCORES = [
    4.10, 2.75, 3.17, 4.95,  # H: Sincerity, Fairness, Greed Avoidance, Modesty
    6.58, 3.85, 3.57, 4.69,  # E: Fearfulness, Anxiety, Dependence, Sentimentality
    4.90, 3.23, 4.27, 5.31,  # X: Social Self-esteem, Social Boldness, Sociability, Liveliness
    4.07, 5.74, 4.60, 4.24,  # A: Forgiveness, Gentleness, Flexibility, Patience
    4.45, 4.98, 5.33, 4.75,  # C: Organization, Diligence, Perfectionism, Prudence
    4.18, 3.73, 3.58, 3.59   # O: Aesthetic Appreciation, Inquisitiveness, Creativity, Unconventionality
]

if __name__ == "__main__":
    my_workstyles = predict_workstyles(MY_HEXACO_SCORES)
    print("--- PREDICTED O*NET WORK STYLES (GEMINI LLM-AS-EXPERT) ---")
    for ws, sc in my_workstyles.items():
        print(f"  {ws:25}: {sc:.2f}")