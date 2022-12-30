#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import streamlit as st
from plotly import graph_objs as go
import nltk
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

nltk.download('vader_lexicon')

reddit = praw.Reddit(
    client_id="ZVb-XoRjJKrevskHU97pIA",
    client_secret="WsmaqLFX32sdgm-0X9rZ4HB6IiL16Q",
    username='mynameisreefer',
    password='passwort123',
    user_agent="social_stocks")

st.title("Social Stocks")

default_stock_goes_here = "GME"
user_stock = st.text_input("Search for a Stock", default_stock_goes_here)

all_subreddits = reddit.subreddit('all')
df = pd.DataFrame(columns=['title', 'date'])
comments = set()

# Search all subreddits using the search function
for submission in all_subreddits.search(query=user_stock, limit=1000):
    comments.add((submission.title, datetime.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d'), submission.subreddit.display_name))

df = pd.DataFrame(comments)
df = df.set_axis(['title', 'date', 'subreddit'], axis=1, inplace=False)

st.write(f"Number of comments about "{default_stock_goes_here}" that have been analyzed: {len(comments)}")
st.write(f"Date of the oldest comment: {df['date'].min()}")

st.dataframe(df[:10])

analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = 0.1
df['sentiment_sum'] = ""
for i in range(len(df)):
    df['sentiment_sum'][i] = analyzer.polarity_scores((df['title'][i]))

for i in range(len(df)):
    df['sentiment'][i] = df['sentiment_sum'][i].get('compound')

df2 = df[['title', 'sentiment']]

stock_sentiment = df['sentiment'].mean()

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    align='center',
    value=stock_sentiment,
    domain={'x': [0, 1], 'y': [0, 1]},
    delta={'reference': 0, 'position': "top"},
    title={'text': "Reddit Sentiment Score"},
    gauge={
        'axis': {'range': [-1, 1]},
        'bar': {
            'color': 'red' if stock_sentiment < 0 else 'green'
        }
    }))

st.plotly_chart(fig)
st.markdown(
    f"""

        The Reddit Sentiment Score represents how this Social Media platform is feeling about your stock.
        It is between -1 (highly negative sentiment) and 1 (highly positive sentiment), while 0 is beeing neutral.

    """
)
