import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if openai_api_key is None:
    st.error("The environment variable OPENAI_API_KEY is not set.")
else:
    st.title("FinanceBot: News Analysis & Research Tool 📈")
    st.sidebar.title("News Article URLs")
    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "D:/Alma better Project 2/Finacial Analyst LLM Chatbot/faiss_store_openai.pkl"

    main_placeholder = st.empty()
    llm = OpenAI(api_key=openai_api_key, model="davinci", temperature=0.9, max_tokens=500)

    def fetch_and_parse_url(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return "\n".join([p.get_text() for p in soup.find_all('p')])
        except requests.RequestException as e:
            st.error(f"Error fetching {url}: {e}")
            return ""

    if process_url_clicked:
        all_texts = [fetch_and_parse_url(url) for url in urls if url]
        combined_text = "\n".join(all_texts)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(combined_text)
        df = pd.DataFrame({'text': texts})
        loader = DataFrameLoader(df, page_content_column="text")

        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        st.write("Loaded Data:", data)  # Debug statement

        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        st.write("Documents Split:", docs)  # Debug statement

        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                st.write("Result Object:", result)  # Debug statement

                st.header("Answer")
                st.write(result.get("answer", "No answer found"))

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    for source in sources.split("\n"):
                        st.write(source)