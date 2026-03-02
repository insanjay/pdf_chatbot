import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vectorstore
def get_vectorstore():

    if os.path.exists("vectorstore/index.faiss"):

        return FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )

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


# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


# Main function
def ask_question(question):

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever()

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
Answer using this context:

{context}

Question: {question}
"""

    response = llm.invoke(prompt)

    return response.content