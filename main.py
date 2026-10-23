import sys
from pathlib import Path

# Ensure UTF-8 output encoding across all terminals (Windows cp1252 fix)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure src module can be imported cleanly
src_dir = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_dir))

from mapping_LLM import predict_workstyles, MY_HEXACO_SCORES

def recommend_jobs(hexaco_scores, top_k=10):
    """
    End-to-end career recommendation pipeline:
    Maps 24 standardized HEXACO facet scores to 21 O*NET Work Styles,
    then computes cosine similarity against 882 O*NET occupations.
    """
    csv_path = Path(__file__).resolve().parent / "data" / "onet_workstyles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Database not found at {csv_path}!")
        
    df_jobs = pd.read_csv(csv_path)
    feature_cols = df_jobs.columns[2:].tolist()
    job_matrix = df_jobs[feature_cols].values

    # 1. Predict candidate's O*NET work style percentiles
    user_workstyles = predict_workstyles(hexaco_scores)
    user_vector = np.array([user_workstyles[col] for col in feature_cols]).reshape(1, -1)

    # 2. Compute profile alignment via Cosine Similarity
    similarities = cosine_similarity(user_vector, job_matrix).flatten()
    df_jobs['match_score'] = np.round(similarities * 100, 2)

    # 3. Retrieve top K matching occupations
    top_jobs = df_jobs.sort_values(by='match_score', ascending=False).head(top_k)
    return user_workstyles, top_jobs

def main():
    print("\n" + "=" * 76)
    print("      HEXACO-O*NET CAREER RECOMMENDER SYSTEM")
    print("      (Psychometric Cross-Domain Alignment via LLM-as-an-Expert)")
    print("=" * 76)

    # Evaluate using candidate's real HEXACO assessment scores
    scores = MY_HEXACO_SCORES
    user_styles, top_jobs = recommend_jobs(scores, top_k=10)

    # 1. Display Candidate Work Styles Summary
    sorted_styles = sorted(user_styles.items(), key=lambda x: x[1], reverse=True)
    print("\n[+] CANDIDATE WORK STYLE PROFILE (O*NET 0-100 NORM CDF SCALE):")
    print("-" * 76)
    print("  * Top Signature Strengths:")
    for style, score in sorted_styles[:3]:
        style_title = style.replace("_", " ").title()
        print(f"     • {style_title:<26}: {score:.1f}/100")

    # 2. Display Top Recommended Occupations
    print("\n[*] TOP 10 RECOMMENDED OCCUPATIONS:")
    print("-" * 76)
    print(f"  {'Rank':<6} {'O*NET Code':<14} {'Occupation Title':<42} {'Match Score'}")
    print("-" * 76)
    for idx, (_, row) in enumerate(top_jobs.iterrows(), 1):
        print(f"  #{idx:<5} {row['job_code']:<14} {row['job_title']:<42} {row['match_score']}%")
    print("=" * 76 + "\n")

if __name__ == "__main__":
    main()
