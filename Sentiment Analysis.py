import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # For interactive plots

st.set_page_config(page_title="Enhanced Sentiment Analysis", layout="wide")
st.title("📝 Enhanced Sentiment Analysis with Visualization")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

model = load_model()

def map_label(label):
    mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    return mapping.get(label, "Unknown")

theme = st.selectbox("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    custom_css = """
    [data-testid="stAppViewContainer"] {background-color: #1e1e1e; color: white;}
    [data-testid="stHeader"] {background-color: #1e1e1e; color: white;}
    .stTextArea textarea {background-color: #424242; color: white; border: 1px solid #90caf9;}
    h1 {color: white;}
    div.stButton > button {background-color: #424242; color: white;}
    div[data-baseweb="select"] > div {background-color: #424242; color: white;}
    """
else:
    custom_css = """
    [data-testid="stAppViewContainer"] {background-color: #e0f7fa; color: black;}
    [data-testid="stHeader"] {background-color: #e0f7fa; color: black;}
    .stTextArea textarea {background-color: #ffffff; color: black; border: 1px solid #90caf9;}
    h1 {color: #0d47a1;}
    div.stButton > button {background-color: #2196F3; color: white;}
    div[data-baseweb="select"] > div {background-color: #ffffff; color: black;}
    """
st.markdown(custom_css, unsafe_allow_html=True)

input_text = st.text_area("Enter text (one sentence per line):", height=200)

if st.button("Analyze Sentiment"):
    if input_text.strip():
        with st.spinner("Analyzing..."):
            sentences = [s.strip() for s in input_text.split('\n') if s.strip()]
            results = model(sentences)
            data = []
            sentiment_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
            short_names = []
            for i, sentence in enumerate(sentences):
                sentiment = map_label(results[i]['label'])
                score = results[i]['score']
                sentiment_counts[sentiment] += 1
                short_label = f"S{i+1}"
                short_names.append(short_label)
                data.append({"Sentence": sentence, "Short": short_label, "Sentiment": sentiment, "Confidence": f"{score:.2%}", "Score": score})
            df = pd.DataFrame(data)

            st.markdown("### 📊 Sentiment Results:")
            st.dataframe(df[["Sentence", "Sentiment", "Confidence"]].style.applymap(
                lambda x: "color: green;" if "Positive" in x else ("color: red;" if "Negative" in x else "color: gray;"),
                subset=["Sentiment"]
            ))
            st.markdown("#### 📌 Note: 'Short' represents a simplified label (S1, S2, etc.) for each sentence, used for clear and compact visualization.")

            st.markdown("### 📊 Sentiment Distribution:")
            fig_pie = px.pie(df, names="Sentiment", title="Sentiment Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### 📊 Sentiment Scores Comparison:")
            fig_bar = px.bar(df, x="Short", y="Score", title="Sentiment Scores Comparison", labels={"Score": "Confidence Score"})
            st.plotly_chart(fig_bar, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "sentiment_analysis_results.csv", "text/csv")
    else:
        st.warning("⚠️ Please enter some text.")
