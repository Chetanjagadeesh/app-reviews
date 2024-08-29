import streamlit as st
st.title("App Reviews Research: Understanding User Feedback and Sentiment")
st.write("Hello, welcome to my first Streamlit app!")

# Text input
App = st.text_input("Enter the google play store app URL to scrape the reviews:")


# Button
if st.button("Submit"):
    st.write(f"The {App} submitted")


# Sidebar content
st.sidebar.title("Leveraging User Insights for Product Innovation")

st.sidebar.write(
    "User reviews offer valuable insights into what works well and what doesn’t. By carefully examining this feedback, product managers can uncover key strengths and weaknesses of their products, allowing them to enhance successful features and address any pain points to improve overall user satisfaction"
)

