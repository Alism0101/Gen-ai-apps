import streamlit as st
from transformers import pipeline

def main():
    # Set the title and description
    st.title(" Sentiment Analysis")
    st.markdown("""
    Analyze the sentiment of your text using a powerful AI model. 
    This app uses the **DistilBERT** model fine-tuned on sentiment analysis to predict whether your text is positive or negative.
    """)
    input_text = st.text_area("Enter your text to analyze:", height=150)
    model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    if st.button("Analyze Sentiment"):
        if input_text.strip():
            with st.spinner("Analyzing sentiment... ⏳"):
                result = model(input_text)
                label = result[0]['label']
                score = result[0]['score']
                st.markdown(f"###  Prediction: **{label}**")
                st.markdown(f"#### Confidence Score: **{score:.2%}**")
        else:
            st.warning("⚠ Please enter some text to analyze.")

if __name__ == "__main__":
    main()
