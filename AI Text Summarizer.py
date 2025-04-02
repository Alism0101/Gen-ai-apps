import streamlit as st
from transformers import pipeline

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Streamlit UI
st.title("📝 AI Text Summarizer")
st.write("Enter a long text below, and get a concise summary!")

# Text Input
long_text = st.text_area("Enter text to summarize:", height=200)

# Summary Parameters
max_length = st.slider("Max Summary Length", min_value=100, max_value=300, value=130)
min_length = st.slider("Min Summary Length", min_value=20, max_value=50, value=30)

# Summarization Button
if st.button("Summarize"):
    if long_text.strip():
        with st.spinner("Generating summary... ⏳"):
            summary = summarizer(long_text, max_length=max_length, min_length=min_length, do_sample=True)
            st.subheader("Summary:")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("⚠ Please enter some text to summarize.")

# Footer
st.markdown("---")
st.markdown("🔗 Built with [Hugging Face Transformers](https://huggingface.co/facebook/bart-large-cnn) & Streamlit 🚀")
