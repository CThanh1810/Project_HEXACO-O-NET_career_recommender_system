import sys
import argparse
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

from mapping_LLM import predict_workstyles, MY_HEXACO_SCORES, WORKSTYLE_NAMES
from vision_parser import extract_hexaco_from_image, select_image_via_dialog

def recommend_jobs(hexaco_scores, top_k=10, zone_filter=None, family_filter=None):
    """
    End-to-end career recommendation pipeline with multi-criteria filtering:
    Maps 24 standardized HEXACO facet scores to 21 O*NET Work Styles,
    filters by Job Zone (Education level) and Job Family (Industry),
    then computes cosine similarity against matching O*NET occupations.
    """
    csv_path = Path(__file__).resolve().parent / "data" / "onet_workstyles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Database not found at {csv_path}!")
        
    df_jobs = pd.read_csv(csv_path)

    # 1. Apply multi-criteria filters if specified
    if zone_filter is not None:
        df_jobs = df_jobs[df_jobs['job_zone'] == zone_filter]
        if df_jobs.empty:
            raise ValueError(f"No occupations found matching Job Zone: {zone_filter}")

    if family_filter is not None:
        df_jobs = df_jobs[df_jobs['job_family'].str.contains(family_filter, case=False, na=False)]
        if df_jobs.empty:
            raise ValueError(f"No occupations found matching Job Family: {family_filter}")

    feature_cols = [col for col in WORKSTYLE_NAMES if col in df_jobs.columns]
    job_matrix = df_jobs[feature_cols].values

    # 2. Predict candidate's O*NET work style percentiles
    user_workstyles = predict_workstyles(hexaco_scores)
    user_vector = np.array([user_workstyles[col] for col in feature_cols]).reshape(1, -1)

    # 3. Compute profile alignment via Cosine Similarity
    similarities = cosine_similarity(user_vector, job_matrix).flatten()
    df_jobs['match_score'] = np.round(similarities * 100, 2)

    # 4. Retrieve top K matching occupations
    top_jobs = df_jobs.sort_values(by='match_score', ascending=False).head(top_k)
    return user_workstyles, top_jobs

def print_results(user_styles, top_jobs, zone_filter=None, family_filter=None):
    sorted_styles = sorted(user_styles.items(), key=lambda x: x[1], reverse=True)
    print("\n[+] CANDIDATE WORK STYLE PROFILE (O*NET 0-100 NORM CDF SCALE):")
    print("-" * 88)
    print("  * Top Signature Strengths:")
    for style, score in sorted_styles[:3]:
        style_title = style.replace("_", " ").title()
        print(f"     • {style_title:<26}: {score:.1f}/100")

    filter_info = []
    if zone_filter:
        filter_info.append(f"Job Zone = {zone_filter}")
    if family_filter:
        filter_info.append(f"Job Family contains '{family_filter}'")
    filter_msg = f" (Filtered by: {', '.join(filter_info)})" if filter_info else ""

    print(f"\n[*] TOP {len(top_jobs)} RECOMMENDED OCCUPATIONS{filter_msg}:")
    print("-" * 88)
    print(f"  {'Rank':<5} {'O*NET Code':<13} {'Occupation Title':<38} {'Education / Zone':<18} {'Match'}")
    print("-" * 88)
    for idx, (_, row) in enumerate(top_jobs.iterrows(), 1):
        zone_label = f"Zone {row['job_zone']}"
        print(f"  #{idx:<4} {row['job_code']:<13} {row['job_title']:<38} {zone_label:<18} {row['match_score']}%")
    print("=" * 88 + "\n")

def main():
    parser = argparse.ArgumentParser(description="HEXACO-O*NET Career Recommender System")
    parser.add_argument("--image", type=str, help="Path to HEXACO result chart image file")
    parser.add_argument("--sample", action="store_true", help="Run with default sample scores")
    parser.add_argument("--zone", type=int, choices=[1, 2, 3, 4, 5], help="Filter by O*NET Job Zone (1-5)")
    parser.add_argument("--family", type=str, help="Filter by Job Family keyword (e.g. Computer, Business, Management)")
    args = parser.parse_args()

    print("\n" + "=" * 88)
    print("      HEXACO-O*NET CAREER RECOMMENDER SYSTEM")
    print("      (Psychometric Cross-Domain Alignment via LLM-as-an-Expert)")
    print("=" * 88)
    print("💡 Haven't taken the assessment yet? Take the official test here:")
    print("   👉 https://hexaco.org (Take a screenshot of your result chart!)")
    print("-" * 88)

    scores = None

    # Case 1: Provided via command line argument --image
    if args.image:
        print(f"🔍 Scanning HEXACO scores from image: {args.image} via Gemini Vision...")
        scores = extract_hexaco_from_image(args.image)
        print("✅ Successfully extracted 24 facet scores!")
    
    # Case 2: Provided via command line argument --sample
    elif args.sample:
        scores = MY_HEXACO_SCORES

    # Case 3: Interactive CLI selection
    else:
        print("Choose input mode:")
        print("  [1] Run with sample / default profile")
        print("  [2] Select result screenshot (Opens Windows File Dialog 🪟)")
        print("  [3] Enter 24 scores manually from keyboard")
        choice = input("Select an option (1/2/3) [Default: 1]: ").strip()

        if choice == "2":
            print("🪟 Opening file dialog... Please select your image file in the popup window.")
            img_path = select_image_via_dialog()
            if not img_path:
                print("No file selected via dialog. You can type or drag-and-drop the path here:")
                img_path = input("Image path: ").strip().strip('"').strip("'")
            
            if img_path:
                print(f"🔍 Scanning image via Gemini Vision: {img_path} ...")
                scores = extract_hexaco_from_image(img_path)
                print("✅ Successfully extracted 24 facet scores!")
            else:
                print("⚠️ No image provided, falling back to sample profile.")
                scores = MY_HEXACO_SCORES

        elif choice == "3":
            print("Enter 24 scores separated by spaces (e.g., 4.1 2.75 3.17 ...):")
            raw_input = input("Scores: ").strip()
            scores = [float(x) for x in raw_input.split()]
            if len(scores) != 24:
                raise ValueError(f"Expected 24 scores, but received {len(scores)}!")
        else:
            scores = MY_HEXACO_SCORES

    # Run recommendation engine with optional filters
    user_styles, top_jobs = recommend_jobs(
        scores, 
        top_k=10, 
        zone_filter=args.zone, 
        family_filter=args.family
    )
    print_results(user_styles, top_jobs, zone_filter=args.zone, family_filter=args.family)

if __name__ == "__main__":
    main()
