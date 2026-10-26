import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure module paths
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
sys.path.append(str(src_dir))

from mapping_LLM import predict_workstyles, MY_HEXACO_SCORES, WORKSTYLE_NAMES
from vision_parser import extract_hexaco_from_bytes

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="HEXACO-O*NET Career Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 24 standard facets mapping
FACETS_INFO = [
    ("Honesty-Humility (H)", [
        ("Sincerity", "Tendency to be genuine in interpersonal relations"),
        ("Fairness", "Tendency to avoid fraud and corruption"),
        ("Greed Avoidance", "Uninterested in lavish wealth and luxury goods"),
        ("Modesty", "Modest and unassuming, viewing oneself as ordinary")
    ]),
    ("Emotionality (E)", [
        ("Fearfulness", "Inclination to avoid physical harm and danger"),
        ("Anxiety", "Tendency to worry and feel stress in difficulties"),
        ("Dependence", "Need for emotional support and comfort from others"),
        ("Sentimentality", "Strong emotional bonds and empathic sensitivity")
    ]),
    ("Extraversion (X)", [
        ("Social Self-esteem", "Positive self-regard in social contexts"),
        ("Social Boldness", "Comfort and confidence in leadership and public settings"),
        ("Sociability", "Enjoyment of conversation, social interaction, and parties"),
        ("Liveliness", "Typical enthusiasm, optimism, and dynamic energy")
    ]),
    ("Agreeableness (A)", [
        ("Forgiveness", "Willingness to trust again and re-establish friendly relations"),
        ("Gentleness", "Mild and lenient dealings, reluctant to judge harshly"),
        ("Flexibility", "Willingness to compromise and cooperate with others"),
        ("Patience", "High threshold for feeling or expressing anger")
    ]),
    ("Conscientiousness (C)", [
        ("Organization", "Tendency to seek order, cleanliness, and structure"),
        ("Diligence", "Strong work ethic, disciplined and motivated to achieve"),
        ("Perfectionism", "Thoroughness, detail-oriented, checks carefully for mistakes"),
        ("Prudence", "Deliberates carefully and inhibits impulsive reactions")
    ]),
    ("Openness to Experience (O)", [
        ("Aesthetic Appreciation", "Enjoyment of beauty in art and nature"),
        ("Inquisitiveness", "Curiosity about the natural and social sciences"),
        ("Creativity", "Preference for innovation, original thoughts, and problem solving"),
        ("Unconventionality", "Receptive to unusual, eccentric, or radical ideas")
    ])
]

# Educational Job Zone definitions
JOB_ZONE_OPTIONS = {
    "All": ("Tất cả các bậc", "All Education Levels"),
    1: ("Zone 1: Phổ thông / Không cần bằng (High School or less)", "Zone 1: Little/No Prep (High School or less)"),
    2: ("Zone 2: Tốt nghiệp Cấp 3 / Chứng chỉ nghề (High School diploma)", "Zone 2: Some Prep (High School diploma)"),
    3: ("Zone 3: Cao đẳng / Trường nghề (Associate / Vocational)", "Zone 3: Medium Prep (Associate degree / Vocational)"),
    4: ("Zone 4: Bằng Đại học (Bachelor's Degree)", "Zone 4: Considerable Prep (Bachelor's degree)"),
    5: ("Zone 5: Sau Đại học / Bác sĩ / Thạc sĩ (Master's / Ph.D. / M.D.)", "Zone 5: Extensive Prep (Master's / Ph.D. / M.D.)")
}

@st.cache_data
def load_onet_data():
    csv_path = root_dir / "data" / "onet_workstyles.csv"
    if not csv_path.exists():
        st.error(f"Data file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)

df_jobs = load_onet_data()

# Initialize session state for the 24 scores
if "hexaco_scores" not in st.session_state:
    st.session_state.hexaco_scores = list(MY_HEXACO_SCORES)

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/O*NET%2028.0-Multi--Criteria%20Matching-success?style=for-the-badge", use_container_width=True)
    st.markdown("## ⚙️ Configuration")
    
    lang = st.radio("🌐 Language / Ngôn ngữ", ["Tiếng Việt", "English"], horizontal=True)
    is_vi = (lang == "Tiếng Việt")
    
    top_k = st.slider(
        "Top Occupations / Số nghề hiển thị:" if not is_vi else "Số lượng nghề nghiệp gợi ý:",
        min_value=5, max_value=25, value=10, step=1
    )

    st.markdown("### 🎓 " + ("Bộ Lọc Hướng Nghiệp Thực Tế" if is_vi else "Multi-Criteria Career Filters"))
    
    # 1. Job Zone (Education level) filter
    zone_choice_keys = list(JOB_ZONE_OPTIONS.keys())
    zone_choice_labels = [JOB_ZONE_OPTIONS[k][0 if is_vi else 1] for k in zone_choice_keys]
    
    selected_zone_label = st.selectbox(
        "Trình độ học vấn mong muốn:" if is_vi else "Minimum Education Level (Job Zone):",
        options=zone_choice_labels,
        index=0
    )
    selected_zone = zone_choice_keys[zone_choice_labels.index(selected_zone_label)]

    # 2. Industry / Job Family filter
    all_families = sorted(df_jobs["job_family"].dropna().unique().tolist())
    selected_families = st.multiselect(
        "Khối ngành nghề quan tâm:" if is_vi else "Target Industry / Job Families:",
        options=all_families,
        default=[],
        placeholder="Tất cả các ngành (All Industries)" if is_vi else "All Industries (Default)"
    )
    
    st.divider()
    st.markdown(
        """
        ### 📖 Hướng dẫn
        1. **Làm bài test**: [hexaco.org](https://hexaco.org)
        2. **Chụp màn hình**: Lưu ảnh biểu đồ kết quả.
        3. **Tải ảnh lên**: Gemini Vision sẽ tự động đọc 24 điểm!
        """ if is_vi else
        """
        ### 📖 Instructions
        1. **Take assessment**: [hexaco.org](https://hexaco.org)
        2. **Capture screenshot** of your result chart.
        3. **Upload image**: Gemini Vision extracts your scores!
        """
    )
    st.caption("Powered by Google Gemini & O*NET 28.0 Database • MIT License")

# -------------------------------------------------------------
# MAIN CONTENT HEADER
# -------------------------------------------------------------
st.title("🧭 HEXACO-O*NET Career Recommender")
st.markdown(
    "**Hệ thống Hướng nghiệp Cá nhân hóa Đa tiêu chí** kết hợp Tâm lý học Tính cách HEXACO với Khung năng lực Nghề nghiệp O*NET."
    if is_vi else
    "**Multi-Criteria Career Alignment System** bridging the HEXACO Personality Model with U.S. O*NET Occupational Taxonomy."
)

st.info(
    "💡 **Bạn chưa có điểm test?** Hãy dành 10-15 phút làm bài kiểm tra miễn phí chính thức tại 👉 [hexaco.org](https://hexaco.org). Sau đó hãy **chụp màn hình biểu đồ kết quả** và tải lên đây!"
    if is_vi else
    "💡 **Haven't taken the assessment yet?** Take the official free test at 👉 [hexaco.org](https://hexaco.org). Once completed, simply **take a screenshot of your result chart** and upload it below!"
)

# -------------------------------------------------------------
# INPUT MODES: TABS
# -------------------------------------------------------------
tab_upload, tab_manual = st.tabs([
    "📸 Tải ảnh kết quả (AI quét tự động)" if is_vi else "📸 Upload Result Chart (AI Auto-Scan)",
    "🎛️ Tự chỉnh thanh trượt (24 Facets)" if is_vi else "🎛️ Interactive Sliders (24 Facets)"
])

# TAB 1: UPLOAD IMAGE
with tab_upload:
    st.markdown(
        "Tải lên ảnh chụp màn hình biểu đồ kết quả từ `hexaco.org`. Công nghệ **Gemini Vision** sẽ tự động trích xuất toàn bộ 24 điểm khía cạnh chỉ trong vài giây."
        if is_vi else
        "Upload your result chart screenshot from `hexaco.org`. **Gemini Vision** will automatically extract all 24 facet scores in seconds."
    )
    
    uploaded_file = st.file_uploader(
        "Chọn file ảnh (.png, .jpg, .jpeg, .webp):" if is_vi else "Choose image file (.png, .jpg, .jpeg, .webp):",
        type=["png", "jpg", "jpeg", "webp"]
    )
    
    col_u1, col_u2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        with col_u1:
            st.image(uploaded_file, caption="Ảnh bạn đã tải lên" if is_vi else "Uploaded Screenshot", use_container_width=True)
            
        with col_u2:
            st.success("✅ Ảnh đã sẵn sàng để quét!" if is_vi else "✅ Image ready for analysis!")
            if st.button("🔍 Quét điểm ngay bằng Gemini Vision" if is_vi else "🔍 Scan Scores via Gemini Vision", type="primary", use_container_width=True):
                with st.spinner("Đang sử dụng Gemini Vision nhận diện biểu đồ..." if is_vi else "Analyzing chart with Gemini Vision..."):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        suffix = Path(uploaded_file.name).suffix.lower()
                        mime_type = "image/png" if suffix == ".png" else "image/jpeg"
                        
                        extracted = extract_hexaco_from_bytes(file_bytes, mime_type=mime_type)
                        st.session_state.hexaco_scores = extracted
                        st.success("🎉 Đã trích xuất thành công 24 điểm! Dữ liệu đã được nạp vào hệ thống." if is_vi else "🎉 Successfully extracted all 24 scores! Loaded into system.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi khi quét ảnh: {e}" if is_vi else f"Error parsing image: {e}")

# TAB 2: MANUAL SLIDERS
with tab_manual:
    st.markdown(
        "Bạn có thể xem hoặc tùy chỉnh điểm số của từng khía cạnh (thang điểm chuẩn hóa dân số, điểm trung bình là 5.0):"
        if is_vi else
        "Inspect or manually adjust your standardized scores (Population mean is 5.0, range ~1.0 to 9.0):"
    )
    
    col_left, col_right = st.columns(2)
    new_scores = list(st.session_state.hexaco_scores)
    global_idx = 0
    
    for domain_idx, (domain_name, facets) in enumerate(FACETS_INFO):
        target_col = col_left if domain_idx < 3 else col_right
        with target_col:
            with st.expander(f"**{domain_name}**", expanded=True):
                for facet_name, facet_desc in facets:
                    curr_val = float(new_scores[global_idx])
                    val = st.slider(
                        f"{facet_name}",
                        min_value=1.0, max_value=8.5, value=curr_val, step=0.01,
                        help=facet_desc,
                        key=f"slider_{global_idx}"
                    )
                    new_scores[global_idx] = val
                    global_idx += 1
                    
    st.session_state.hexaco_scores = new_scores

# -------------------------------------------------------------
# COMPUTATION & RECOMMENDATION RESULTS
# -------------------------------------------------------------
st.divider()

if st.button("🚀 GỢI Ý NGHỀ NGHIỆP PHÙ HỢP" if is_vi else "🚀 RECOMMEND MATCHING CAREERS", type="primary", use_container_width=True):
    current_scores = st.session_state.hexaco_scores
    
    # 1. Filter dataset by Job Zone & Job Family
    df_filtered = df_jobs.copy()
    
    if selected_zone != "All":
        df_filtered = df_filtered[df_filtered["job_zone"] == selected_zone]
        
    if selected_families:
        df_filtered = df_filtered[df_filtered["job_family"].isin(selected_families)]
        
    if df_filtered.empty:
        st.warning("⚠️ Không tìm thấy nghề nghiệp nào thỏa mãn bộ lọc hiện tại. Vui lòng mở rộng lựa chọn ngành hoặc trình độ!" if is_vi else "⚠️ No occupations match your selected filters. Please broaden your industry or education choices!")
    else:
        # 2. Predict Workstyles
        predicted_workstyles = predict_workstyles(current_scores)
        feature_cols = [col for col in WORKSTYLE_NAMES if col in df_filtered.columns]
        job_matrix = df_filtered[feature_cols].values
        
        user_vector = np.array([predicted_workstyles[col] for col in feature_cols]).reshape(1, -1)
        similarities = cosine_similarity(user_vector, job_matrix).flatten()
        
        df_filtered["match_score"] = np.round(similarities * 100, 2)
        top_matches = df_filtered.sort_values(by="match_score", ascending=False).head(top_k)
        
        # ---------------------------------------------------------
        # DISPLAY METRICS & PROFILE
        # ---------------------------------------------------------
        st.markdown("## 📊 " + ("Hồ Sơ Phong Cách Làm Việc (O*NET 0-100)" if is_vi else "Candidate Work Style Profile (O*NET 0-100)"))
        
        sorted_ws = sorted(predicted_workstyles.items(), key=lambda x: x[1], reverse=True)
        
        st.markdown("### ⭐ " + ("Top 3 Thế Mạnh Nổi Bật Nhất:" if is_vi else "Top 3 Signature Strengths:"))
        m_cols = st.columns(3)
        for i, (ws_k, ws_v) in enumerate(sorted_ws[:3]):
            with m_cols[i]:
                title = ws_k.replace("_", " ").title()
                st.metric(label=f"#{i+1} {title}", value=f"{ws_v:.1f} / 100")
                
        # Bar chart of 21 work styles
        with st.expander("📈 " + ("Xem toàn bộ 21 chỉ số phong cách làm việc" if is_vi else "View all 21 Work Style percentiles"), expanded=False):
            df_ws_chart = pd.DataFrame({
                "Work Style": [k.replace("_", " ").title() for k in predicted_workstyles.keys()],
                "Percentile Score (0-100)": list(predicted_workstyles.values())
            }).sort_values(by="Percentile Score (0-100)", ascending=True)
            
            st.bar_chart(df_ws_chart.set_index("Work Style"), horizontal=True, color="#2563EB")
            
        # ---------------------------------------------------------
        # DISPLAY OCCUPATION RECOMMENDATIONS
        # ---------------------------------------------------------
        filter_summary = []
        if selected_zone != "All":
            filter_summary.append(f"Zone {selected_zone}")
        if selected_families:
            filter_summary.append(", ".join(selected_families))
        filter_str = f" ({', '.join(filter_summary)})" if filter_summary else ""

        st.markdown(f"## 🎯 " + (f"Top {len(top_matches)} Nghề Nghiệp Phù Hợp Nhất{filter_str}" if is_vi else f"Top {len(top_matches)} Recommended Careers{filter_str}"))
        
        res_table = []
        for rank, (_, row) in enumerate(top_matches.iterrows(), 1):
            res_table.append({
                "Rank" if not is_vi else "Hạng": f"#{rank}",
                "O*NET Code" if not is_vi else "Mã O*NET": row["job_code"],
                "Occupation Title" if not is_vi else "Tên Nghề Nghiệp": row["job_title"],
                "Job Family" if not is_vi else "Khối Ngành": row["job_family"],
                "Education Level" if not is_vi else "Học Vấn": f"Zone {row['job_zone']}",
                "Match Score (%)" if not is_vi else "Độ Khớp (%)": f"{row['match_score']}%"
            })
        
        st.dataframe(pd.DataFrame(res_table), use_container_width=True, hide_index=True)
        
        # Expanders for top 3 careers detail
        st.markdown("### 🔍 " + ("Phân tích chi tiết Top 3 nghề dẫn đầu:" if is_vi else "In-Depth Insights on Top 3 Occupations:"))
        for rank, (_, row) in enumerate(top_matches.head(3).iterrows(), 1):
            with st.expander(f"#{rank} [{row['job_code']}] {row['job_title']} - " + ("Độ phù hợp: " if is_vi else "Match: ") + f"{row['match_score']}%"):
                st.markdown(f"**Mã O*NET:** `{row['job_code']}` | **Khối ngành:** {row['job_family']} | **Học vấn:** {row['job_zone_title']}")
                job_traits = row[feature_cols].sort_values(ascending=False).head(5)
                st.markdown("**Top 5 đặc tính quan trọng nhất của nghề này:**" if is_vi else "**Top 5 critical work styles for this occupation:**")
                trait_strs = [f"- **{t.replace('_', ' ').title()}**: {val:.1f}/100" for t, val in job_traits.items()]
                st.markdown("\n".join(trait_strs))

st.markdown("---")
st.caption("Developed with ❤️ using Google Gemini & O*NET Career Content Model.")
