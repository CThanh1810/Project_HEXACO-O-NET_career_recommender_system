from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mapping_LLM import predict_workstyles, MY_HEXACO_SCORES # Import hàm tính điểm và bộ điểm bài test của bạn từ mapping_LLM

# 1. Đọc dữ liệu 882 nghề nghiệp O*NET
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "data" / "onet_workstyles.csv"
df_jobs = pd.read_csv(csv_path)

feature_cols = df_jobs.columns[2:].tolist()
job_matrix = df_jobs[feature_cols].values

# 2. Lấy điểm O*NET Work Styles tính tự động từ bài test HEXACO của bạn
user_workstyles = predict_workstyles(MY_HEXACO_SCORES)

# Tạo vector người dùng sắp xếp đúng thứ tự 21 cột của O*NET
user_vector = np.array([user_workstyles[col] for col in feature_cols]).reshape(1, -1)

# 3. Tính độ khớp bằng Cosine Similarity
similarities = cosine_similarity(user_vector, job_matrix).flatten()
df_jobs['match_score'] = np.round(similarities * 100, 2)

# 4. In ra Top 10 nghề nghiệp phù hợp nhất
top_10 = df_jobs.sort_values(by='match_score', ascending=False).head(10)

print("\n🎯 TOP 10 RECOMMENDED OCCUPATIONS BASED ON YOUR HEXACO PROFILE:")
print("=" * 76)
for idx, (_, row) in enumerate(top_10.iterrows(), 1):
    print(f"{idx:2d}. [{row['job_code']}] {row['job_title']:<48} (Match: {row['match_score']}%)")
print("=" * 75)