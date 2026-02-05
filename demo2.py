import streamlit as st
import joblib
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Nepali News Classifier",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
    <style>
    .prediction-box {
        background-color: #161b22;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b; /* Red accent like a pin */
        font-size: 20px;
        font-weight: 500;
        margin-top: -10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)




# Title and description
st.markdown('<h1 class="main-title">📰 Nepali News Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Classify Nepali news articles into categories using AI</p>', unsafe_allow_html=True)

# Load model with error handling
model = joblib.load("nepali_news_classifier.joblib")

# Information section
with st.expander("ℹ️ About this app"):
    st.write("""
    **How to use:**
    1. Paste or type your Nepali news article in the text box below
    2. Click the "Classify" button
    3. Get instant category prediction
    
    **Supported Categories:**
    - Information Technology (SuchanaPrabidhi)
    - Entertainment (Manoranjan)
    - Diaspora (Prabas)
    - Opinion / Thoughts (Bichar)
    - Economy (ArthaBanijya)
    - Nation / Country (Desh)
    - Health (Swasthya)
    - Sports (Khelkud)
    - World / International (Bishwa)
    - Literature (Sahitya)
    
    **Tips for best results:**
    - Enter at least 50-100 characters
    - Use complete sentences
    - Ensure text is in Nepali (Devanagari script)
    """)

# Divider
st.markdown("---")

# Text input section
st.markdown("### 📝 Enter News Article")
input_text = st.text_area(
    label='Enter Nepali news text:',
    placeholder='यहाँ आफ्नो समाचार पेस्ट गर्नुहोस्...',
    max_chars=2000,
    height=200,
    help="Maximum 2000 characters"
)

# Show character count
if input_text:
    char_count = len(input_text)
    word_count = len(input_text.split())
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"📊 Characters: {char_count}/2000")
    with col2:
        st.caption(f"📝 Words: {word_count}")

# Classify button
if st.button("🔍 Classify News", type="primary", use_container_width=True):
    if not input_text or len(input_text.strip()) < 10:
        st.warning("⚠️ Please enter at least 10 characters of text to classify!")
    else:
        # Show spinner while processing
        with st.spinner('Analyzing text...'):
            time.sleep(0.5)  # Brief delay for better UX
            
            try:
                # Get prediction
                output = model.predict([input_text])
                predicted_category = output[0]
                
                # Try to get probabilities if available
                try:
                    probabilities = model.predict_proba([input_text])[0]
                    classes = model.classes_
                    
                    # Display main prediction
                    st.markdown("### 🎯 Prediction Result")
                    st.markdown(f'<div class="prediction-box">📌 {predicted_category}</div>', 
                              unsafe_allow_html=True)
                    
                    # Display confidence scores
                    st.markdown("### 📊 Confidence Scores")
                    
                    # Create a dataframe for better visualization
                    import pandas as pd
                    prob_data = pd.DataFrame({
                        'Category': classes,
                        'Confidence': probabilities * 100
                    }).sort_values('Confidence', ascending=False)
                    
                    # Show top 3 predictions
                    for idx, row in prob_data.head(3).iterrows():
                        confidence = row['Confidence']
                        category = row['Category']
                        
                        # Color based on confidence
                        if confidence > 50:
                            color = "#4CAF50"  # Green
                        elif confidence > 20:
                            color = "#3700FF"  # Orange
                        else:
                            color = "#F44336"  # Red
                        
                        st.markdown(f"""
                        <div style="background-color: {color}20; padding: 10px; 
                                    border-left: 4px solid {color}; margin: 5px 0; border-radius: 5px;">
                            <strong>{category}</strong>: {confidence:.2f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show bar chart
                    st.bar_chart(prob_data.set_index('Category')['Confidence'])
                    
                except AttributeError:
                    # Model doesn't have predict_proba (e.g., some pipelines)
                    st.markdown("### 🎯 Prediction Result")
                    st.markdown(f'<div class="prediction-box">📌 {predicted_category}</div>', 
                              unsafe_allow_html=True)
                
               
                
            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")

# Divider
st.markdown("---")

# Example section
with st.expander("💡 See Examples"):
    st.markdown("""
    **Example 1 - Sports:**
    ```
    नेपालमा आज फुटबल खेल भएको छ। खेलमा नेपाली टोलीले राम्रो प्रदर्शन गरेको थियो।
    ```
    
    **Example 2 - Technology:**
    ```
    नयाँ स्मार्टफोन बजारमा आएको छ। यो फोनमा उन्नत प्रविधि र राम्रो क्यामेरा रहेको छ।
    ```
    
    """)
