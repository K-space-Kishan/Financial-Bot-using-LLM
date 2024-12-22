from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set Hugging Face API Key
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Ensure the API key is set
if huggingface_api_key is None:
    st.error("The environment variable HUGGINGFACE_API_KEY is not set.")
else:
    st.title("FinanceBot: News Analysis & Research Tool 📈")
    st.sidebar.title("News Article URLs")
    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "D:/Alma better Project 2/Finacial Analyst LLM Chatbot/faiss_store_huggingface.pkl"

    main_placeholder = st.empty()

    def fetch_and_parse_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return "\n".join([p.get_text() for p in soup.find_all('p')])
        except requests.RequestException as e:
            st.error(f"Error fetching {url}: {e}")
            return ""

    def query_gpt_j(prompt):
        """Query the Hugging Face GPT-J API."""
        API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
        headers = {"Authorization": f"Bearer {huggingface_api_key}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()

    if process_url_clicked:
        all_texts = [fetch_and_parse_url(url) for url in urls if url]
        combined_text = "\n".join(all_texts)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(combined_text)
        df = pd.DataFrame({'text': texts})
        loader = DataFrameLoader(df, page_content_column="text")

        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        docs = text_splitter.split_documents(data)

        st.write("Debug docs:", docs)  # Debugging doc structure

        # Initialize Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        vectorstore = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                retriever = vectorstore.as_retriever()
                relevant_docs = retriever.get_relevant_documents(query)
                st.write("Relevant Documents:", relevant_docs)

                # Use GPT-J API to generate an answer
                prompt = f"Based on the following documents, answer the question:\n{relevant_docs}\n\nQuestion: {query}\nAnswer:"
                result = query_gpt_j(prompt)
                st.write("Result Object:", result)

                st.header("Answer")
                if isinstance(result, list) and result:
                    st.write(result[0].get("generated_text", "No answer found"))
                else:
                    st.write("No answer found or unexpected result format")
