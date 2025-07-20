import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Constants
DOCUMENTS_PATH = "./documents"
FAISS_INDEX_PATH = "faiss_index"

def create_vector_store():
    """
    Loads documents from the DOCUMENTS_PATH, splits them into chunks, 
    creates embeddings using Gemini, and saves them to a FAISS vector store.
    """
    print("Starting vector store creation/update...")

    if not os.path.exists(DOCUMENTS_PATH) or not os.listdir(DOCUMENTS_PATH):
        print(f"The directory '{DOCUMENTS_PATH}' is empty. No vector store will be created.")
        return

    loader = DirectoryLoader(DOCUMENTS_PATH, glob='**/*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    
    if not documents:
        print(f"No PDF documents could be loaded from '{DOCUMENTS_PATH}'. Aborting.")
        return

    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    print("Cleaning document chunks...")
    for doc in docs:
        doc.page_content = doc.page_content.encode('utf-8', 'replace').decode('utf-8')
    print("Cleaning complete.")

    print("Creating embeddings with Google Gemini. This may take a while...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = FAISS.from_documents(docs, embeddings)
    print("Vector store created.")

    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"Vector store saved to '{FAISS_INDEX_PATH}'")

# This allows the script to be run directly if needed, for initial setup.
if __name__ == "__main__":
    create_vector_store()