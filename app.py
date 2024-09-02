# import streamlit as st
# import pandas as pd
# from review_scraper import get_app_reviews_dataframe, Sort
# from data_preprocessing import clean_dataframe, extract_app_id
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma

# # Initialize session state for clean_data
# if 'clean_data' not in st.session_state:
#     st.session_state.clean_data = None

# # Function to create embeddings
# def create_embeddings(dataframe, api_key):
#     embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#     # Assuming each row in the DataFrame can be represented as a single string
#     texts = dataframe.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
#     return texts, embeddings.embed_documents(texts)

# # Function to set up the vector store
# def setup_vector_store(dataframe, api_key):
#     texts, embeddings = create_embeddings(dataframe, api_key)
#     vector_store = Chroma.from_texts(texts, embeddings)
#     return vector_store.as_retriever()

# # Function to create the conversational retrieval chain
# def make_chain(dataframe, api_key):
#     model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=api_key)
#     retriever = setup_vector_store(dataframe, api_key)
#     chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
#     return chain

# # Streamlit UI
# st.title("App Reviews Research: Understanding User Feedback and Sentiment")

# # Text input for app URL
# App = st.text_input("Enter the Google Play Store app URL to scrape the reviews:")

# if st.button("Fetch Reviews"):
#     app_id = extract_app_id(App)
#     if app_id:
#         st.write(f"Fetching reviews for {app_id}...")
        
#         df = get_app_reviews_dataframe(
#             app_id,
#             reviews_count=25000,
#             lang='en',
#             country='in',
#             sort=Sort.NEWEST,
#             sleep_milliseconds=100
#         )
        
#         st.write(f"Total reviews fetched: {len(df)}")
        
#         average_score = df['score'].mean()
#         st.write(f"\nAverage Rating: {average_score:.2f}")

#         # Clean the data and store it in session state
#         st.session_state.clean_data = clean_dataframe(df)

#         # Option to download the full DataFrame
#         csv = st.session_state.clean_data[['reviewid', 'content', 'score', 'appversion']].to_csv(index=False)
#         st.download_button(
#             label="Download full data as CSV",
#             data=csv,
#             file_name=f"{app_id}_reviews.csv",
#             mime="text/csv",
#         )
#     else:
#         st.error("Please enter a valid app ID")

# # User input for API key (masked)
# user_api_key = st.text_input("Enter your OpenAI API Key (will be masked):", type="password")

# # User input for question
# question = st.text_input("Ask your question:")

# if st.button("Submit"):
#     if question and user_api_key:
#         if st.session_state.clean_data is not None:  # Check if clean_data is defined
#             chain = make_chain(st.session_state.clean_data, user_api_key)  # Use the user-provided API key
#             response = chain({"question": question, "chat_history": ""})
#             st.write(f"Chatbot response: {response['answer']}")
#         else:
#             st.warning("Please fetch reviews first.")
#     else:
#         st.warning("Please enter both your API key and a question.")

import streamlit as st
import pandas as pd
from review_scraper import get_app_reviews_dataframe, Sort
from data_preprocessing import clean_dataframe, extract_app_id
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

# Initialize session state
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chain' not in st.session_state:
    st.session_state.chain = None

# Function to create embeddings
def create_embeddings(dataframe, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    texts = dataframe.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    return texts, embeddings

# Function to set up the vector store
def setup_vector_store(dataframe, api_key):
    texts, embeddings = create_embeddings(dataframe, api_key)
    vector_store = Chroma.from_texts(texts, embeddings)
    return vector_store.as_retriever()

# Function to create the conversational retrieval chain
def make_chain(dataframe, api_key):
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=api_key)
    retriever = setup_vector_store(dataframe, api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever, memory=memory)
    return chain

# Streamlit UI
st.title("App Reviews Research: Understanding User Feedback and Sentiment")

# Text input for app URL
App = st.text_input("Enter the Google Play Store app URL to scrape the reviews:")

if st.button("Fetch Reviews"):
    app_id = extract_app_id(App)
    if app_id:
        st.write(f"Fetching reviews for {app_id}...")
        df = get_app_reviews_dataframe(
            app_id,
            reviews_count=2000,
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            sleep_milliseconds=100
        )
        st.write(f"Total reviews fetched: {len(df)}")
        average_score = df['score'].mean()
        st.write(f"\nAverage Rating: {average_score:.2f}")
        
        # Clean the data and store it in session state
        st.session_state.clean_data = clean_dataframe(df)
        
        # Option to download the full DataFrame
        csv = st.session_state.clean_data[['reviewid', 'content', 'score', 'appversion']].to_csv(index=False)
        st.download_button(
            label="Download full data as CSV",
            data=csv,
            file_name=f"{app_id}_reviews.csv",
            mime="text/csv",
        )
    else:
        st.error("Please enter a valid app ID")

# User input for API key (masked)
user_api_key = st.text_input("Enter your OpenAI API Key (will be masked):", type="password")

if user_api_key and st.session_state.clean_data is not None:
    if st.session_state.chain is None:
        st.session_state.chain = make_chain(st.session_state.clean_data, user_api_key)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for question
if question := st.chat_input("Ask your question about the reviews:"):
    if st.session_state.chain is not None:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.chat_message("assistant"):
            response = st.session_state.chain({"question": question})
            st.markdown(response['answer'])
        st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
    else:
        st.warning("Please fetch reviews and enter your API key first.")
# import streamlit as st
# import pandas as pd
# from review_scraper import get_app_reviews_dataframe, Sort
# from data_preprocessing import clean_dataframe, extract_app_id
# from together import Together

# # Initialize session state for clean_data
# if 'clean_data' not in st.session_state:
#     st.session_state.clean_data = None

# # Function to create embeddings (if Together AI provides this functionality)
# def create_embeddings(dataframe):
#     texts = dataframe.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
#     return texts

# # Function to set up the vector store (if needed for Together AI)
# def setup_vector_store(dataframe, api_key):
#     texts = create_embeddings(dataframe)
#     # Assuming Together AI provides a way to create embeddings
#     # This is a placeholder for actual Together AI embedding logic
#     embeddings = []  # Replace with actual Together AI embedding logic
#     return embeddings

# # Function to create the conversational retrieval chain using Together AI
# def make_chain(dataframe, api_key):
#     client = Together(api_key=api_key)
#     # Assuming Together AI has a similar chain setup
#     # This is a placeholder for actual Together AI chain logic
#     return client

# # Streamlit UI
# st.title("App Reviews Research: Understanding User Feedback and Sentiment")

# # Text input for app URL
# App = st.text_input("Enter the Google Play Store app URL to scrape the reviews:")

# if st.button("Fetch Reviews"):
#     app_id = extract_app_id(App)
#     if app_id:
#         st.write(f"Fetching reviews for {app_id}...")
#         df = get_app_reviews_dataframe(
#             app_id,
#             reviews_count=25000,
#             lang='en',
#             country='in',
#             sort=Sort.NEWEST,
#             sleep_milliseconds=100
#         )
#         st.write(f"Total reviews fetched: {len(df)}")
#         average_score = df['score'].mean()
#         st.write(f"\nAverage Rating: {average_score:.2f}")

#         # Clean the data and store it in session state
#         st.session_state.clean_data = clean_dataframe(df)

#         # Option to download the full DataFrame
#         csv = st.session_state.clean_data[['reviewid', 'content', 'score', 'appversion']].to_csv(index=False)
#         st.download_button(
#             label="Download full data as CSV",
#             data=csv,
#             file_name=f"{app_id}_reviews.csv",
#             mime="text/csv",
#         )
#     else:
#         st.error("Please enter a valid app ID")

# # User input for Together AI API key (masked)
# user_api_key = st.text_input("Enter your Together AI API Key (will be masked):", type="password")

# # User input for question
# question = st.text_input("Ask your question:")

# if st.button("Submit"):
#     if question and user_api_key:
#         if st.session_state.clean_data is not None:  # Check if clean_data is defined
#             client = Together(api_key=user_api_key)
#             response = client.chat.completions.create(
#                 model="meta-llama/Llama-3-70b-chat-hf",  # Example model
#                 messages=[{"role": "user", "content": question}]
#             )
#             st.write(f"Chatbot response: {response.choices[0].message.content}")
#         else:
#             st.warning("Please fetch reviews first.")
#     else:
#         st.warning("Please enter both your API key and a question.")