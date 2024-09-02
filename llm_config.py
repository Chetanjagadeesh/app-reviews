import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def create_embeddings(dataframe):
    embeddings = OpenAIEmbeddings()
    # Assuming each row in the DataFrame can be represented as a single string
    texts = dataframe.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    return texts, embeddings.embed_documents(texts)

# Step 2: Set up the vector store
def setup_vector_store(dataframe):
    texts, embeddings = create_embeddings(dataframe)
    vector_store = Chroma.from_texts(texts, embeddings)
    return vector_store.as_retriever()



def make_chain(dataframe, api_key):
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0,openai_api_key=api_key)
    # Define your retrieval method here (e.g., using embeddings)
    retriever = setup_vector_store(dataframe)   # Set up your retriever based on the DataFrame
    chain = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    return chain