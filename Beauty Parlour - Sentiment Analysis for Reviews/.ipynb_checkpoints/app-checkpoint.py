
# Your Streamlit code here...
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("radhika_reviews_with_sentiment.csv")

st.set_page_config(page_title="Radhika Beauty Parlour Reviews", layout="wide")

# Title
st.title("💄 Radhika Beauty Parlour - Google Review Sentiment Dashboard")

# Sentiment Summary
st.subheader("📊 Sentiment Distribution")

sentiment_counts = df["sentiment"].value_counts()
fig, ax = plt.subplots()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
plt.xticks(rotation=0)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
st.pyplot(fig)

# Optional: Average rating if you scraped numeric rating
if "rating" in df.columns:
    st.subheader("⭐ Average Rating")
    avg_rating = df["rating"].mean()
    st.metric(label="Average Rating", value=f"{avg_rating:.2f} / 5.0")

# Filter Section
st.subheader("🔍 Browse Reviews by Sentiment")
selected_sentiment = st.selectbox("Choose sentiment to view:", ["All", "Positive", "Negative", "Neutral"])

if selected_sentiment != "All":
    filtered_df = df[df["sentiment"] == selected_sentiment]
else:
    filtered_df = df

# Display Reviews
st.dataframe(filtered_df[["name", "review", "sentiment"]], use_container_width=True)
