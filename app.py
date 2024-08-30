import streamlit as st
import re
import altair as alt
from review_scraper import get_app_reviews_dataframe, Sort
from data_preprocessing import clean_dataframe , extract_app_id


st.title("App Reviews Research: Understanding User Feedback and Sentiment")

# Text input
App = st.text_input("Enter the google play store app URL to scrape the reviews:")

# app_id=extract_app_id(App)

if st.button("Fetch Reviews"):
 if app_id:
  app_id=extract_app_id(App)
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
            
  # Option to download the full DataFrame
  csv = df.to_csv(index=False)
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



data = get_app_reviews_dataframe(
        app_id,
        reviews_count=25000,
        lang='en',
        country='in',
        sort=Sort.NEWEST,
        sleep_milliseconds=100  # Add a small delay between requests
    )

clean_data= clean_dataframe(data)


# Count occurrences of each rating
rating_counts = clean_data['score'].value_counts().reset_index()
rating_counts.columns = ['score', 'count']
rating_counts = rating_counts.sort_values(by='score')  # Sort by rating value

# Create an Altair bar chart
chart = alt.Chart(rating_counts).mark_bar().encode(
    x=alt.X('score:O', title='Rating'),
    y=alt.Y('count:Q', title='Count'),
    tooltip=['score', 'count']  # Add tooltips for interactivity
).properties(
    title='Rating Count Chart'
)

# Display the chart in Streamlit
st.altair_chart(chart, use_container_width=True)
