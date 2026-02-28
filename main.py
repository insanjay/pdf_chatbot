import os
from dotenv import load_dotenv

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# LLM
from langchain_groq import ChatGroq


# -------------------------
# Load ENV
# -------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")


# -------------------------
# Load PDF
# -------------------------

loader = PyPDFLoader("pdf\Sanjay Kumar.pdf")

documents = loader.load()


# -------------------------
# Split Text
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)


# -------------------------
# Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# Vector Store
# -------------------------

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("vectorstore") # saves the data

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


# -------------------------
# Chat Loop (Manual RAG)
# -------------------------

print("\nPDF Chatbot Ready. Type exit to quit.\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break


    # retrieve context
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])


    # create prompt
    prompt = f"""
Answer based only on this context:

{context}

Question: {query}
"""


    # generate answer
    response = llm.invoke(prompt)


    print("\nBot:", response.content)
    print("\n")