import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Social Media Sentiment Dashboard",
    layout="wide"
)

st.title("Social Media Sentiment Analysis Dashboard")
st.markdown("**Emotion-Aware Monitoring System for Social Media Posts (Reddit Dataset)**")

# ----------------------------------
# LOAD DATASET
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("depression_dataset_reddit_cleaned.csv")

df = load_data()
TEXT_COLUMN = df.columns[0]

# ----------------------------------
# LOAD NLP MODELS
# ----------------------------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis")
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None   # allow all emotions
    )
    return sentiment_model, emotion_model

sentiment_pipeline, emotion_pipeline = load_models()

# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================
st.sidebar.header("Dashboard Controls")

sample_size = st.sidebar.slider(
    "Number of Dataset Posts to Analyze",
    min_value=10,
    max_value=200,
    value=50
)

run_analysis = st.sidebar.button("Run Analysis")

# ----------------------------------
# SESSION STATE INIT
# ----------------------------------
if "df_sample" not in st.session_state:
    st.session_state.df_sample = None

# ----------------------------------
# RUN DATASET ANALYSIS ONLY WHEN BUTTON CLICKED
# ----------------------------------
if run_analysis:
    df_sample = df.sample(sample_size, random_state=42)

    sentiments, emotions, emotion_scores = [], [], []

    with st.spinner("Running NLP analysis on dataset..."):
        for text in df_sample[TEXT_COLUMN]:
            text = str(text)[:512]
            sent_result = sentiment_pipeline(text)[0]
            emo_result = emotion_pipeline(text)

            # highest emotion
            top_emotion = max(emo_result[0], key=lambda x: x["score"])

            sentiments.append(sent_result["label"])
            emotions.append(top_emotion["label"])
            emotion_scores.append(top_emotion["score"])

    df_sample["Sentiment"] = sentiments
    df_sample["Emotion"] = emotions
    df_sample["Emotion Confidence"] = emotion_scores

    st.session_state.df_sample = df_sample

# ----------------------------------
# DISPLAY RESULTS (ONLY AFTER RUN)
# ----------------------------------
if st.session_state.df_sample is not None:

    df_sample = st.session_state.df_sample

    # ----------------------------------
    # DASHBOARD VISUALIZATION
    # ----------------------------------
    st.subheader("Analytics Dashboard")

    col1, col2 = st.columns(2)

    sentiment_count = df_sample["Sentiment"].value_counts().reset_index()
    sentiment_count.columns = ["Sentiment", "Count"]

    fig_sentiment = px.pie(
        sentiment_count,
        names="Sentiment",
        values="Count",
        title="Sentiment Distribution"
    )
    col1.plotly_chart(fig_sentiment, use_container_width=True)

    emotion_count = df_sample["Emotion"].value_counts().reset_index()
    emotion_count.columns = ["Emotion", "Count"]

    fig_emotion = px.bar(
        emotion_count,
        x="Emotion",
        y="Count",
        title="Emotion Distribution"
    )
    col2.plotly_chart(fig_emotion, use_container_width=True)

    # ----------------------------------
    # TOP CONFIDENT EMOTIONAL POSTS
    # ----------------------------------
    st.subheader("Most Confident Emotion Predictions")

    top_emotion_posts = df_sample.sort_values(
        by="Emotion Confidence",
        ascending=False
    ).head(10)

    st.dataframe(
        top_emotion_posts[[TEXT_COLUMN, "Sentiment", "Emotion", "Emotion Confidence"]],
        use_container_width=True
    )

    # ----------------------------------
    # DATA PREVIEW
    # ----------------------------------
    st.subheader("Processed Data Preview")

    st.dataframe(
        df_sample[[TEXT_COLUMN, "Sentiment", "Emotion", "Emotion Confidence"]].head(20),
        use_container_width=True
    )

    # ==========================================================
    # LIVE SENTIMENT & EMOTION ANALYSIS (REAL-TIME DEMO)
    # ==========================================================
    st.markdown("---")
    st.subheader("Live Sentiment & Emotion Analysis")

    user_input = st.text_area(
        "Enter social media text:",
        height=150,
        placeholder="Example: I feel very stressed and overwhelmed today..."
    )

    if st.button("Analyze Text") and user_input.strip() != "":
        with st.spinner("Analyzing text..."):
            text = user_input[:512]

            sent_result = sentiment_pipeline(text)[0]
            emo_results = emotion_pipeline(text)[0]  # all emotions

        # Convert to DataFrame
        emo_df = pd.DataFrame(emo_results)
        emo_df = emo_df.sort_values("score", ascending=False)

        top_emotion = emo_df.iloc[0]

        # -------------------------------
        # METRICS
        # -------------------------------
        col1, col2 = st.columns(2)
        col1.metric("Sentiment Polarity", sent_result["label"])
        col2.metric(
            "Primary Emotion",
            f"{top_emotion['label']} ({top_emotion['score']:.2f})"
        )

        # -------------------------------
        # MULTI-EMOTION CONFIDENCE GRAPH
        # -------------------------------
        st.markdown("### Emotion Distribution for This Text")

        fig_conf = px.bar(
            emo_df,
            x="label",
            y="score",
            title="Emotion Confidence Scores",
            labels={"label": "Emotion", "score": "Confidence"},
        )

        st.plotly_chart(fig_conf, use_container_width=True)

        # -------------------------------
        # ANALYZED TEXT
        # -------------------------------
        st.markdown("### Analyzed Text")
        st.write(user_input)
