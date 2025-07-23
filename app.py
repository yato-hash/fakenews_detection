import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set up the page title and a brief description
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("Fake News Detector ")
st.write(
    "Enter the text of a news article below to classify it as Real or Fake."
)

# --- Text Input Area ---
st.header("Enter News Text")
news_text = st.text_area("Paste the article text here:", height=250)

# --- Prediction Button ---
if st.button("Classify News", type="primary"):
    if news_text:
        try:
            # Create instances of the prediction pipeline
            data = CustomData(text=news_text)
            pred_df = data.get_data_as_frame()
            
            pipeline = PredictPipeline()
            
            # Show a spinner while predicting
            with st.spinner('Analyzing the news...'):
                results = pipeline.predict(pred_df)

            # --- Display Results ---
            st.header("Analysis Result")
            if results[0] == 0:
                st.success(" This news appears to be REAL.")
            else:
                st.error(" This news appears to be FAKE.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter some text to analyze.")