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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize session state
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chain' not in st.session_state:
    st.session_state.chain = None

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    return tokenizer, model

def create_embeddings(dataframe):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = dataframe.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    embeddings = model.encode(texts)
    return texts, embeddings

def setup_vector_store(dataframe):
    texts, embeddings = create_embeddings(dataframe)
    vector_store = Chroma.from_texts(texts, HuggingFaceEmbeddings())
    return vector_store.as_retriever()

def make_chain(dataframe):
    tokenizer, model = load_model()
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    retriever = setup_vector_store(dataframe)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    return chain

def display_review_stats(df):
    st.subheader("Review Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        average_score = df['score'].mean()
        st.metric("Average Rating", f"{average_score:.2f}")
    with col3:
        positive_reviews = len(df[df['score'] >= 4])
        st.metric("Positive Reviews (%)", f"{(positive_reviews / len(df) * 100):.1f}%")

    # Rating distribution
    st.subheader("Rating Distribution")
    rating_counts = df['score'].value_counts().sort_index()
    st.bar_chart(rating_counts)

# Streamlit UI
st.set_page_config(page_title="App Review Analyzer", layout="wide")
st.title("App Reviews Research: Understanding User Feedback and Sentiment")

# Sidebar for app URL input and review fetching
with st.sidebar:
    st.header("Fetch App Reviews")
    App = st.text_input("Enter the Google Play Store app URL:")
    num_reviews = st.number_input("Number of reviews to fetch:", min_value=100, max_value=25000, value=1000, step=100)
    if st.button("Fetch Reviews"):
        app_id = extract_app_id(App)
        if app_id:
            with st.spinner(f"Fetching reviews for {app_id}..."):
                df = get_app_reviews_dataframe(
                    app_id,
                    reviews_count=num_reviews,
                    lang='en',
                    country='in',
                    sort=Sort.NEWEST,
                    sleep_milliseconds=100
                )
            st.success(f"Total reviews fetched: {len(df)}")
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
            st.error("Please enter a valid app URL")

# Main content area
if st.session_state.clean_data is not None:
    display_review_stats(st.session_state.clean_data)

    # Main chat interface
    st.header("Chat with your App Reviews")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for question
    if question := st.chat_input("Ask your question about the reviews:"):
        if st.session_state.chain is None:
            with st.spinner("Initializing AI model..."):
                st.session_state.chain = make_chain(st.session_state.clean_data)
            if st.session_state.chain:
                st.success("AI model initialized and ready!")

        if st.session_state.chain is not None:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chain({"question": question})
                    st.markdown(response['answer'])
                    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
        else:
            st.warning("Please fetch reviews first.")

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
else:
    st.info("Please fetch reviews using the sidebar to start analyzing.")