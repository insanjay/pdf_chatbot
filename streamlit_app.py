import os
import streamlit as st
from dotenv import load_dotenv

from rag_core import ask_question
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# -----------------------
# ENV
# -----------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="PDF Chatbot")

st.title("📄 PDF RAG Chatbot")


# -----------------------
# Load Embeddings
# -----------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# -----------------------
# Load or Create Vectorstore
# -----------------------

@st.cache_resource
def load_vectorstore():

    if os.path.exists("vectorstore/index.faiss"):

        return FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        loader = PyPDFLoader("pdf/sample.pdf")

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vs = FAISS.from_documents(chunks, embeddings)

        vs.save_local("vectorstore")

        return vs


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever()


# -----------------------
# Load LLM
# -----------------------

@st.cache_resource
def load_llm():

    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )


llm = load_llm()


# -----------------------
# Chat Memory
# -----------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# display history

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------
# User Input
# -----------------------

if prompt := st.chat_input("Ask about the PDF"):


    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )


    with st.chat_message("user"):

        st.markdown(prompt)


    answer = ask_question(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)