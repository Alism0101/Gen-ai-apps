import streamlit as st
from transformers import pipeline

def main():
    st.title("Hugging Face Model Demo")

    # Create an input text box
    input_text = st.text_input("Enter your text", "")

    model = pipeline("sentiment-analysis", model = "distilbert-base-uncased-finetuned-sst-2-english")

    # Create a button to trigger model inference
    if st.button("Analyze"):
        # Perform inference using the loaded model
        result = model(input_text)
        st.write("Prediction:", result[0]['label'], "| Score:", result[0]['score'])

if __name__ == "__main__":
    main()
