import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Sentiment Analysis", layout="wide")
st.title(" Sentiment Analysis with Visualization")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

model = load_model()

def map_label(label):
    label_mapping = {
        "LABEL_0": "Very Negative",
        "LABEL_1": "Negative",
        "LABEL_2": "Neutral",
        "LABEL_3": "Positive",
        "LABEL_4": "Very Positive"
    }
    return label_mapping.get(label, "Unknown")

theme = st.selectbox("Select Theme", ["Light", "Dark"])
st.markdown(f"<style>body {{ background-color: {'#1e1e1e' if theme == 'Dark' else '#f0f0f0'}; color: {'white' if theme == 'Dark' else 'black'}; }}</style>", unsafe_allow_html=True)

input_text = st.text_area("Enter text (one sentence per line):", height=200)
if st.button("Analyze Sentiment"):
    if input_text.strip():
        with st.spinner("Analyzing..."):
            sentences = [s.strip() for s in input_text.split('\n') if s.strip()]
            results = model(sentences)

            data = []
            sentiment_counts = {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0}

            for i, sentence in enumerate(sentences):
                sentiment = map_label(results[i]['label'])
                score = results[i]['score']
                sentiment_counts[sentiment] += 1
                data.append({"Sentence": sentence, "Sentiment": sentiment, "Confidence": f"{score:.2%}", "Score": score})

            df = pd.DataFrame(data)

            st.markdown("###  Sentiment Results:")
            st.dataframe(df.style.applymap(
                lambda x: "color: green;" if "Positive" in x else ("color: red;" if "Negative" in x else "color: gray;"),
                subset=["Sentiment"]
            ))

            fig1, ax1 = plt.subplots()
            labels = list(sentiment_counts.keys())
            sizes = list(sentiment_counts.values())
            ax1.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#FF6B6B", "#FFA07A", "#FFD700", "#98FB98", "#32CD32"])
            ax1.set_title("Sentiment Distribution")
            st.pyplot(fig1)

            st.markdown("###  Sentiment Scores Comparison:")
            fig2, ax2 = plt.subplots()
            df['Score'] = df['Score'].astype(float)
            df.plot(kind="barh", x="Sentence", y="Score", color="#4682B4", ax=ax2, legend=False)
            ax2.set_xlabel("Confidence Score")
            ax2.set_title("Sentiment Scores for Each Sentence")
            st.pyplot(fig2)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "sentiment_analysis_results.csv", "text/csv")
    else:
        st.warning("⚠ Please enter some text.")
