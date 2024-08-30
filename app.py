import streamlit as st
import re
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from review_scraper import get_app_reviews_dataframe, Sort
from data_preprocessing import clean_dataframe , extract_app_id



st.title("App Reviews Research: Understanding User Feedback and Sentiment")

# Text input
App = st.text_input("Enter the google play store app URL to scrape the reviews:")

# app_id=extract_app_id(App)

if st.button("Fetch Reviews"):
 app_id=extract_app_id(App)
 if app_id:
  st.write(f"Fetching reviews for {app_id}...")
            
  df = get_app_reviews_dataframe(
                app_id,
                reviews_count=25000,
                lang='en',
                country='in',
                sort=Sort.NEWEST,
                sleep_milliseconds=100
            )
            
  st.write(f"Total reviews fetched: {len(df)}")
            
  average_score = df['score'].mean()
  st.write(f"\nAverage Rating: {average_score:.2f}")

  clean_data= clean_dataframe(df)

  # Option to download the full DataFrame
  csv = clean_data[['reviewid','content','score','appversion']].to_csv(index=False)
  st.download_button(
      label="Download full data as CSV",
      data=csv,
      file_name=f"{app_id}_reviews.csv",
      mime="text/csv",
  )
else:
   st.error("Please enter an app ID")

# Sidebar content
st.sidebar.title("Leveraging User Insights for Product Innovation")

st.sidebar.write(
    "User reviews offer valuable insights into what works well and what doesn’t. By carefully examining this feedback, product managers can uncover key strengths and weaknesses of their products, allowing them to enhance successful features and address any pain points to improve overall user satisfaction"
)

st.sidebar.write("---")

st.sidebar.write(
    "To learn more about how this project was done, visit [our website](https://www.yourwebsite.com)."
)



score_counts = clean_data['score'].value_counts().reset_index()
score_counts.columns = ['score', 'count']

# Check if DataFrame is not empty
if not score_counts.empty:
    # Create a bar chart with Plotly
    fig = px.bar(score_counts, x='score', y='count', title='Rating Distributions')
    fig.update_layout(
        xaxis_title='Ratings',
        yaxis_title='Count'
    )

    # Plot the bar chart
    st.plotly_chart(fig)
else:
   st.write("No data available to display the chart.")

negative_rated_combined_text = clean_data[clean_data['score'] < 3]['content'].str.cat(sep=' ')
positive_rated_combined_text = clean_data[clean_data['score'] > 3]['content'].str.cat(sep=' ')

# Generate word clouds
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_rated_combined_text)
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_rated_combined_text)

# Convert word clouds to images and then to numpy arrays
img_array_neg = np.array(wordcloud_neg.to_image())
img_array_pos = np.array(wordcloud_pos.to_image())

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

# Plot negative sentiment word cloud
with col1:
    fig_neg = go.Figure()
    fig_neg.add_trace(go.Image(z=img_array_neg))
    fig_neg.update_layout(
        title='Reviews Word Cloud for Negative Sentiment',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_neg, use_container_width=True)

# Plot positive sentiment word cloud
with col2:
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Image(z=img_array_pos))
    fig_pos.update_layout(
        title='Reviews Word Cloud for Positive Sentiment',
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_pos, use_container_width=True)

