import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set up the Streamlit app
st.set_page_config(page_title="Real-Time Sentiment Dashboard", layout="wide")

# Title and description
st.title("Real-Time Sentiment Dashboard")
st.markdown("A dashboard to visualize real-time public sentiment from social media.")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('mock_tweets.csv')
    data['created_at'] = pd.to_datetime(data['created_at'])
    return data

data = load_data()

# Sidebar filters
st.sidebar.header("Filters")
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [data['created_at'].min().date(), data['created_at'].max().date()]
)

filtered_data = data[
    (data['created_at'] >= pd.to_datetime(start_date)) & 
    (data['created_at'] <= pd.to_datetime(end_date))
]

keyword = st.sidebar.text_input("Search Keyword", "")
if keyword:
    filtered_data = filtered_data[filtered_data['text'].str.contains(keyword, case=False)]

# Sentiment distribution (Pie Chart)
st.subheader("Sentiment Distribution")
sentiment_counts = filtered_data['sentiment_score'].apply(
    lambda x: "Positive" if x > 0 else "Neutral" if x == 0 else "Negative"
).value_counts()

fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["#63cdda", "#f8a5c2", "#786fa6"])
st.pyplot(fig)

# Sentiment trend over time (Line Chart)
st.subheader("Sentiment Trend Over Time")
sentiment_trend = filtered_data.groupby(filtered_data['created_at'].dt.date)['sentiment_score'].mean()
st.line_chart(sentiment_trend)

# Top users (Bar Chart)
st.subheader("Top Users by Mentions")
filtered_data['mentions'] = filtered_data['mentions'].apply(eval)  # Convert string to list
top_users = filtered_data['mentions'].explode().value_counts().head(10)
st.bar_chart(top_users)

# Word cloud
st.subheader("Word Cloud of Tweets")
all_text = " ".join(filtered_data['text'])
wordcloud = WordCloud(background_color="white").generate(all_text)
st.image(wordcloud.to_array())

# Engagement metrics (Table)
st.subheader("Engagement Metrics")
st.dataframe(
    filtered_data[['text', 'retweet_count', 'reply_count', 'like_count', 'quote_count']].sort_values(by='like_count', ascending=False)
)

# Footer
st.markdown("Created using Streamlit | © 2025")

# Instructions for deployment on GitHub
st.sidebar.info("""
**Deployment Instructions**:
1. Push this file (`app.py`) and the dataset (`mock_tweets.csv`) to a GitHub repository.
2. Deploy using [Streamlit Community Cloud](https://streamlit.io/cloud).
""")
