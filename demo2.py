import streamlit as st
import joblib
from pathlib import Path
import sklearn

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Nepali News Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Nepali News Classifier")

# ✅ FIXED: correct __file__
BASE_DIR = Path(__file__).parent

PIPE_PATH = BASE_DIR / "nepali_news_classifier.joblib"
LE_PATH   = BASE_DIR / "nepali_news_label_encoder.joblib"

@st.cache_resource
def load_artifacts():
    try:
        pipe = joblib.load(PIPE_PATH)
        le = joblib.load(LE_PATH)
        return pipe, le
    except Exception as e:
        st.error("❌ Failed to load model files")
        st.error(f"Error: {e}")
        st.info(f"Python version: {tuple(__import__('sys').version_info)}")
        st.info(f"Scikit-learn version: {sklearn.__version__}")
        st.stop()

pipe, le = load_artifacts()

# -----------------------
# Input
# -----------------------
text = st.text_area(
    "News text (Nepali)",
    height=220,
    placeholder="यहाँ नेपाली समाचार लेख्नुहोस्..."
)

# -----------------------
# Predict
# -----------------------
if st.button("🔍 Classify", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred_num = pipe.predict([text])[0]
        pred_label = le.inverse_transform([pred_num])[0]
        st.success(f"🧾 Category: **{pred_label}**")