import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("🚀 Starting Data Ingestion...")

all_docs = []
data_folder = "./data_files"

for file in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        all_docs.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(file_path)
        all_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(all_docs)

# Cloud Support Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings, 
    persist_directory="./db_storage"
)

print("✅ Success! Database created at ./db_storage")