
import streamlit as st
import joblib

st.title ("Nepali news categroy prediction")
input_text =st.text_input("enter the news you want to predict")

model = joblib.load("nepali_news_classifier.joblib")
if st.button("PREDICT"):
    output = model.predict([input_text])
    st.success(output[0])