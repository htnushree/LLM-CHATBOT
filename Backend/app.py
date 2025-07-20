import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from vector_store_manager import create_vector_store, DOCUMENTS_PATH
from rag_handler import RAGHandler
import traceback # Import traceback for detailed error logging

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# Google API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in the .env file.")

# Global RAG Handler variable
rag_handler = None

def initialize_rag_handler():
    """Loads or reloads the RAG handler."""
    global rag_handler
    try:
        print("Initializing RAG Handler...")
        rag_handler = RAGHandler(faiss_index_path="./faiss_index")
        print("RAG Handler initialized successfully.")
    except Exception as e:
        print(f"Could not initialize RAG Handler: {e}")
        print("Please make sure a vector store exists. You can create one by uploading a PDF.")
        rag_handler = None

# --- API Routes ---
@app.route('/')
def index():
    """A simple route to confirm the server is running."""
    return "Flask backend with PDF upload is running!"

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handles PDF file uploads and triggers re-indexing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

    
    try:
        filename = secure_filename(file.filename)
        # Ensure the documents directory exists
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)
        save_path = os.path.join(DOCUMENTS_PATH, filename)
        file.save(save_path)
        
        print(f"File '{filename}' uploaded successfully. Rebuilding vector store...")
        # Rebuild the vector store with the new file
        create_vector_store()
        
        # Reload the RAG handler to use the new index
        initialize_rag_handler()
        
        return jsonify({"message": f"File '{filename}' uploaded and indexed successfully."}), 200
    except Exception as e:
        # Log the full error to the console for debugging
        print("--- AN ERROR OCCURRED DURING UPLOAD AND PROCESSING ---")
        traceback.print_exc()
        print("----------------------------------------------------")
        # Return a proper JSON error response to the frontend
        return jsonify({"error": "An internal error occurred while processing the PDF. Please check the backend console for details."}), 500
    # --- END OF ERROR HANDLING BLOCK ---



# In Backend/app.py

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles chat messages by passing them to the RAG handler."""
    # The "global rag_handler" line was removed from here
    
    # If the handler isn't loaded in this worker, try to load it.
    if rag_handler is None:
        print("RAG handler not found in this worker, attempting to initialize...")
        initialize_rag_handler()
        # If it's still None after trying, the index doesn't exist yet.
        if rag_handler is None:
            return jsonify({"error": "Chatbot is not ready. Please upload a PDF document first."}), 503

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request. 'message' is required."}), 400

    user_message = data['message']

    try:
        bot_response = rag_handler.get_response(user_message)
        return jsonify({"reply": bot_response})
    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "An error occurred while processing your message."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize the RAG handler on startup
    initialize_rag_handler()
    app.run(debug=True, port=5000)
