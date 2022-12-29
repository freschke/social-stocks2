#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import streamlit as st
from plotly import graph_objs as go
import nltk
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
with open('pw.txt', 'r') as f:
    pw = f.read()

reddit = praw.Reddit(
    client_id="ZVb-XoRjJKrevskHU97pIA",
    client_secret="WsmaqLFX32sdgm-0X9rZ4HB6IiL16Q",
    username='mynameisreefer',
    password=pw,
    user_agent="social_stocks")

st.title("Social Stocks")

default_stock_goes_here = "GME"
user_stock = st.text_input("Search for a Stock", default_stock_goes_here)
default_amount_comments = 1000
user_comment_amount = st.text_input("How many comments to you want to analyze", default_amount_comments)

subreddit = reddit.subreddit("wallstreetbets")
df = pd.DataFrame()
df['title'] = ""
comments = set()
for submission in subreddit.search(user_stock, limit=int(user_comment_amount)):
    comments.add(submission.title)

df = pd.DataFrame(comments)
df = df.set_axis(['title'], axis=1, inplace=False)

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
    title={'text': "Reddit Sentiment"},
    gauge={'axis': {'range': [-1, 1]}}))

st.plotly_chart(fig)
