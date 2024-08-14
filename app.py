import streamlit as st
st.title("My First Streamlit App")
st.write("Hello, welcome to my first Streamlit app!")

# Text input
name = st.text_input("Enter your name:")

# Number input
age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)

# Button
if st.button("Submit"):
    st.write(f"Hello {name}, you are {age} years old!")

