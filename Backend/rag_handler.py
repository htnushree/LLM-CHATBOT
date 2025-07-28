# backend/rag_handler.py

import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

class RAGHandler:
    def __init__(self, faiss_index_path="./faiss_index"):
        """
        Initializes the RAG handler by loading the vector store and setting up the QA chain.
        """
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index path not found: {faiss_index_path}")

        # Use the same embedding model that was used to create the store
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the vector store with the required security flag
        self.vector_store = FAISS.load_local(
            folder_path=faiss_index_path, 
            embeddings=self.embeddings,
            # This flag is now required by LangChain for security
            allow_dangerous_deserialization=True 
        )
        

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=False
        )

    def get_response(self, query: str) -> str:
        """Gets a response from the RAG chain."""
        if not self.qa_chain:
            return "The question-answering chain is not initialized."
        
        result = self.qa_chain.invoke({"query": query})
        return result.get("result", "Sorry, I could not find an answer.")
