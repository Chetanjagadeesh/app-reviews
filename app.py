import streamlit as st
st.title("App Reviews Research: Understanding User Feedback and Sentiment")
st.write("Hello, welcome to my first Streamlit app!")

# Text input
App = st.text_input("Enter the google play store app URL:")

# Button
if st.button("Submit"):
    st.write(f"The {App} submitted")

