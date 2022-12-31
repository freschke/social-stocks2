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
import yfinance as yf

nltk.download('vader_lexicon')

reddit = praw.Reddit(
    client_id="ZVb-XoRjJKrevskHU97pIA",
    client_secret="WsmaqLFX32sdgm-0X9rZ4HB6IiL16Q",
    username='mynameisreefer',
    password='passwort123',
    user_agent="social_stocks")

st.title("Social Stocks")

default_stock_goes_here = "GME"
user_stock = st.text_input("Type in your stock (WKN for visualising the stock price)", default_stock_goes_here)

all_subreddits = reddit.subreddit('all')
df = pd.DataFrame(columns=['title', 'date'])
comments = set()
current_date = datetime.datetime.now()
max_age = datetime.timedelta(days=365*2)

for submission in all_subreddits.search(query=user_stock, limit=1000):
  # Get the creation date of the comment
  creation_date = datetime.datetime.fromtimestamp(submission.created_utc)

  # Calculate the age of the comment
  age = current_date - creation_date

  # If the age of the comment is less than or equal to 2 years
  if age <= max_age:
    # Add the comment to the comments set
    comments.add((submission.title, creation_date.strftime('%Y-%m-%d'), submission.subreddit.display_name))

df = pd.DataFrame(comments)
df = df.set_axis(['title', 'date', 'subreddit'], axis=1, inplace=False)

st.write(f"Number of comments about {user_stock} that have been analyzed: {len(comments)}")
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
# Aggregate the data by date and compute the mean sentiment for each date
sentiment_by_date = df.groupby("date")["sentiment"].mean()

stock_data = yf.download(tickers=user_stock, start=df['date'].min(), end=datetime.datetime.now())
stock_data.reset_index(inplace=True)
stock_data.rename(columns={"Date": "date"}, inplace=True)
stock_data.set_index("date", inplace=True)

# Merge the sentiment data with the stock data
combined_data = pd.concat([sentiment_by_date, stock_data["Close"]], axis=1)

# Create the first trace for the sentiment data
sentiment_plot = go.Scatter(x=combined_data.index, y=combined_data["sentiment"], name="Sentiment")

# Create the second trace for the stock data
stock_plot = go.Scatter(x=combined_data.index, y=combined_data["Close"], name="Stock Price", yaxis="y2")

# Create the layout with two y-axes
layout = go.Layout(
    yaxis=dict(title="Sentiment"),
    yaxis2=dict(title="Stock Price", overlaying="y", side="right")
)

# Create the figure with two traces and the specified layout
figure = go.Figure(data=[sentiment_plot, stock_plot], layout=layout)


# Display the figure using st.plotly_chart

st.plotly_chart(figure)
st.markdown(f"Sentiment of the {len(comments)} comments & the corresponding stock price")


