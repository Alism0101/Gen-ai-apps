import streamlit as st
from transformers import pipeline
import pandas as pd

def main():
    # Set the title and description
    st.title("📝 Advanced Sentiment Analysis")
    st.markdown("""
    Analyze the sentiment of your text using a powerful AI model with enhanced precision. 
    This app uses the **RoBERTa Sentiment Model** to detect nuanced emotions like very positive or very negative sentiments.
    """)

    # Create an input text area for batch input
    input_text = st.text_area("Enter text to analyze (one sentence per line):", height=200)

    # Load the sentiment analysis model with improved accuracy
    @st.cache_resource
    def load_model():
        return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    model = load_model()

    # Create a button to trigger model inference
    if st.button("Analyze Sentiment"):
        if input_text.strip():
            with st.spinner("Analyzing sentiment... ⏳"):
                # Split text into lines for batch processing
                sentences = input_text.split('\n')
                results = model(sentences)

                # Prepare the output
                sentiments = [res['label'] for res in results]
                scores = [res['score'] for res in results]

                # Display the results
                st.markdown("### 📊 Sentiment Analysis Results:")
                for i, sentence in enumerate(sentences):
                    st.markdown(f"**Sentence:** {sentence}")
                    st.markdown(f"- **Sentiment:** {sentiments[i]}")
                    st.markdown(f"- **Confidence:** {scores[i]:.2%}")
                    st.progress(scores[i])

                # Display a summary table
                df = pd.DataFrame(results, index=[f"Sentence {i+1}" for i in range(len(sentences))])
                st.markdown("### 📑 Summary:")
                st.dataframe(df)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

if __name__ == "__main__":
    main()
