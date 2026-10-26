from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mapping_LLM import predict_workstyles, MY_HEXACO_SCORES, WORKSTYLE_NAMES

# 1. Load enriched O*NET dataset
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "data" / "onet_workstyles.csv"
df_jobs = pd.read_csv(csv_path)

feature_cols = [col for col in WORKSTYLE_NAMES if col in df_jobs.columns]
job_matrix = df_jobs[feature_cols].values

# 2. Predict user work styles from real HEXACO test
user_workstyles = predict_workstyles(MY_HEXACO_SCORES)
user_vector = np.array([user_workstyles[col] for col in feature_cols]).reshape(1, -1)

# 3. Compute cosine similarity
similarities = cosine_similarity(user_vector, job_matrix).flatten()
df_jobs['match_score'] = np.round(similarities * 100, 2)

# 4. Display top 10 recommended jobs with Job Zone and Family metadata
top_10 = df_jobs.sort_values(by='match_score', ascending=False).head(10)

print("\n🎯 TOP 10 RECOMMENDED OCCUPATIONS BASED ON YOUR HEXACO PROFILE:")
print("=" * 90)
for idx, (_, row) in enumerate(top_10.iterrows(), 1):
    print(f"{idx:2d}. [{row['job_code']}] {row['job_title']:<42} | Zone {row['job_zone']} | {row['job_family']:<22} (Match: {row['match_score']}%)")
print("=" * 90)
