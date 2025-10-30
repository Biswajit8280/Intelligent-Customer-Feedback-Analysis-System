import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from transformers import pipeline

st.set_page_config(page_title="AI Customer Feedback Analyzer", layout="wide")

st.title(" Intelligent Customer Feedback Analysis System")
st.markdown("Analyze, summarize, and visualize customer feedback using AI")

@st.cache_resource
def load_sentiment_pipeline():
    try:
        sentiment_model = pipeline("sentiment-analysis")
        return sentiment_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

sentiment_pipeline = load_sentiment_pipeline()


st.sidebar.header(" Upload Feedback Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")

    
    st.subheader(" Data Preview")
    st.dataframe(df.head())

    text_col = st.selectbox("Select the column containing feedback text:", df.columns)

    if st.button(" Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            df["Sentiment_Result"] = df[text_col].astype(str).apply(lambda x: sentiment_pipeline(x)[0]['label'])
        st.success("✅ Sentiment analysis completed!")

        st.subheader(" Sentiment Results")
        st.dataframe(df[[text_col, "Sentiment_Result"]].head())

        st.subheader(" Sentiment Distribution")
        fig = px.histogram(df, x="Sentiment_Result", color="Sentiment_Result", title="Sentiment Count")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(" AI Summarization")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        combined_text = " ".join(df[text_col].astype(str).tolist())[:3000]  # limit text length
        summary = summarizer(combined_text, max_length=130, min_length=30, do_sample=False)
        st.write(summary[0]['summary_text'])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download Analyzed Data",
            data=csv,
            file_name="analyzed_feedback.csv",
            mime="text/csv",
        )

else:
    st.info(" Upload a CSV file to start analysis.")
