import streamlit as st
import pandas as pd
import re
import textwrap
import os

from ai_ds_analyst.agent_graph import graph


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# PROFESSIONAL AI ANALYST THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       FONTS
    ======================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');


    /* ========================================================
       GLOBAL PAGE
    ======================================================== */

    html, body {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       HIDE STREAMLIT TOP NAVIGATION BAR
    ======================================================== */

    [data-testid="stHeader"] {
        display: none;
    }

    /* ========================================================
    HEADER
    ======================================================== */

    .dashboard-header {
        background: #173F70;
        border-radius: 0 0 16px 16px;
        border-bottom: 4px solid #F59E0B;
        padding: 32px 36px;
        margin-bottom: 45px;
        box-shadow: 0 5px 15px rgba(15, 23, 42, 0.10);
    }

    .dashboard-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 34px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -1px;
        margin: 0 0 6px 0;
    }

    .dashboard-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        color: #DCE9F7;
        margin: 0;
    }


    /* ========================================================
       SECTION TITLES
    ======================================================== */

    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #173F6F;
        margin-top: 28px;
        margin-bottom: 14px;
        padding-left: 12px;
        border-left: 4px solid #F59E0B;
    }


    /* ========================================================
       UPLOAD CARD
    ======================================================== */

    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 1px solid #D7E1EE;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }


    [data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC;
        border: 2px dashed #AFC3DA;
        border-radius: 10px;
    }


    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F5F9FF;
        border-color: #3B82F6;
    }


    /* ========================================================
       UPLOAD BUTTON
    ======================================================== */

    [data-testid="stFileUploader"] button {
        background-color: #173F6F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }


    [data-testid="stFileUploader"] button:hover {
        background-color: #24558F !important;
    }

    /* ========================================================
    UPLOAD SUCCESS
    ======================================================== */

    .upload-success {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 10px;
        padding: 13px 16px;
        margin: 14px 0 28px 0;
        color: #065F46;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
    }

    .success-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background: #10B981;
        color: white;
        border-radius: 50%;
        font-size: 14px;
        font-weight: 700;
    }
    /* ========================================================
       SUCCESS MESSAGE
    ======================================================== */

    [data-testid="stAlert"] {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 10px;
        color: #166534;
    }


    /* ========================================================
       DATAFRAME
    ======================================================== */

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border: 1px solid #D7E1EE;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }

    /* ========================================================
    DATASET PREVIEW CARD
    ======================================================== */

    .dataset-card {
        background: #FFFFFF;
        border: 1px solid #D9E2EF;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    /* Dataset information line */
    .dataset-info {
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #64748B;
        margin-top: -5px;
        margin-bottom: 12px;
    }

    /* ========================================================
       QUESTION INPUT
    ======================================================== */

    [data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #B8C7D9 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        padding: 13px 14px !important;
    }


    [data-testid="stTextInput"] input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
    }


    [data-testid="stTextInput"] input::placeholder {
        color: #94A3B8 !important;
    }


    /* ========================================================
       LABELS
    ======================================================== */

    label {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }


    /* ========================================================
       ANALYSIS CARD
    ======================================================== */

    .analysis-card {
        background-color: #FFFFFF;
        border: 1px solid #D7E1EE;
        border-left: 5px solid #3B82F6;
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }


    .analysis-label {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 700;
        color: #3B82F6;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }


    .analysis-result {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.75;
        color: #334155;
    }


    /* ========================================================
       DIVIDER
    ======================================================== */

    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 32px 0;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="dashboard-header">
    <div class="dashboard-title">
        📊 AI Data Analyst
    </div>
    <div class="dashboard-subtitle">
        Ask questions about your dataset using natural language.
    </div>
</div>
""",
    unsafe_allow_html=True
)
# ============================================================
# UPLOAD DATASET
# ============================================================

st.markdown(
    '<div class="section-title">Upload Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)


# ============================================================
# DATASET
# ============================================================

if uploaded_file:

    st.markdown(
    f"""
    <div class="upload-success">
        <span class="success-icon">✓</span>
        <span>
            <strong>{uploaded_file.name}</strong>
            uploaded successfully
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
    df = pd.read_csv(uploaded_file)

    st.markdown(
    '<div class="section-title">Dataset Preview</div>',
    unsafe_allow_html=True
)

    st.markdown(
        f"""
        <div class="dataset-info">
            Showing {len(df)} rows × {len(df.columns)} columns
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dataset-card">',
        unsafe_allow_html=True
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=False
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # ========================================================
    # ASK QUESTION
    # ========================================================

    st.markdown(
        '<div class="section-title">Ask Your Dataset</div>',
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Ask a question about your dataset:",
        placeholder="Example: What is the average price?"
    )

    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if question:

        os.makedirs("data", exist_ok=True)

        file_name = os.path.basename(uploaded_file.name)

        dataset_path = os.path.abspath(
            os.path.join("data", file_name)
        )

        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        print("DATASET PATH:", dataset_path)
        print("FILE EXISTS:", os.path.exists(dataset_path))

        with st.spinner("🤖 Analyzing your dataset..."):

            result = graph.invoke(
                {
                    "user_query": question,
                    "dataset_path": dataset_path,
                    "analysis_result": "",
                    "final_answer": "",
                    "messages": [],
                }
            )

        answer = result["final_answer"]

        print("RAW ANSWER:", repr(answer))

        answer = re.sub(r"<[^>]+>", "", answer)
        answer = answer.replace("**", "")
        answer = "\n".join(
            line.strip()
            for line in answer.splitlines()
            if line.strip()
        )

        print("CLEANED ANSWER:", repr(answer))
          # ====================================================
        # ANALYSIS RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">🤖 Analysis</div>',
            unsafe_allow_html=True
        )

        with st.container():
            st.markdown(
                """
                <div class="analysis-card">
                    <div class="analysis-label">
                        RESULT
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(answer)