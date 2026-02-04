# import streamlit as st
# import joblib
# from pathlib import Path

# # -----------------------
# # Page config
# # -----------------------
# st.set_page_config(
#     page_title="Nepali News Classifier",
#     page_icon="📰",
#     layout="centered"
# )

# st.title("📰 Nepali News Classifier")

# BASE_DIR = Path(__file__).parent
# PIPE_PATH = BASE_DIR / "nepali_news_classifier.joblib"
# LE_PATH   = BASE_DIR / "nepali_news_label_encoder.joblib"

# @st.cache(allow_output_mutation=True)
# def load_artifacts():
#     pipe = joblib.load(PIPE_PATH)
#     le = joblib.load(LE_PATH)
#     return pipe, le

# pipe, le = load_artifacts()

# # -----------------------
# # Input
# # -----------------------
# text = st.text_area(
#     "News text (Nepali)",
#     height=220,
#     placeholder="यहाँ नेपाली समाचार लेख्नुहोस्..."
# )

# # -----------------------
# # Predict
# # -----------------------
# if st.button("🔍 Classify"):
#     if not text.strip():
#         st.warning("Please enter some text.")
#     else:
#         # numeric prediction
#         pred_num = pipe.predict([text])[0]

#         # decode to label
#         pred_label = le.inverse_transform([pred_num])[0]

#         st.success(f"🧾 Category: **{pred_label}**")

import streamlit as st
import joblib

st.title("News category prediction")
input_text = st.text_input("Enter the news you want to predict")

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("navive_bayes.joblib")

model = load_model()

if st.button("PREDICT"):
    if input_text.strip():
        output = model.predict([input_text])
        st.success(output[0])
    else:
        st.warning("Please enter some text.")
