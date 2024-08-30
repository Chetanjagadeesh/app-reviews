import streamlit as st
import re
import altair as alt
from review_scraper import run_scraper
from data_preprocessing import clean_dataframe , extract_app_id


st.title("App Reviews Research: Understanding User Feedback and Sentiment")

# Text input
App = st.text_input("Enter the google play store app URL to scrape the reviews:")

app_id=extract_app_id(App)

if st.button("Submit"):
    st.write(f"The submitted App id is {app_id}")


# Sidebar content
st.sidebar.title("Leveraging User Insights for Product Innovation")

st.sidebar.write(
    "User reviews offer valuable insights into what works well and what doesn’t. By carefully examining this feedback, product managers can uncover key strengths and weaknesses of their products, allowing them to enhance successful features and address any pain points to improve overall user satisfaction"
)

st.sidebar.write("---")

st.sidebar.write(
    "To learn more about how this project was done, visit [our website](https://www.yourwebsite.com)."
)



data = review_scraper(app_id)

clean_data= data_preprocessing(data)


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
