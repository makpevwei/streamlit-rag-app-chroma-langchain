"""
================================================================================
RAG APPLICATION WITH LANGCHAIN & CHROMA - BEGINNER FRIENDLY VERSION
================================================================================

This is a Retrieval-Augmented Generation (RAG) application that helps you:
✅ Upload your documents (PDF, Word, Excel, CSV, Text, HTML)
✅ Process them automatically with AI
✅ Ask questions about your documents
✅ Get answers with sources shown

RAG PIPELINE (7 Steps):
========================
Step 1: LOADING     → Load documents from user uploads
Step 2: SPLITTING   → Split large documents into smaller chunks
Step 3: EMBEDDING   → Convert text chunks into AI vectors (embeddings)
Step 4: STORAGE     → Store vectors in Chroma database for fast search
Step 5: RETRIEVAL   → Find relevant chunks when user asks a question
Step 6: PROMPTING   → Create a prompt with the question + relevant context
Step 7: GENERATION  → Use Groq LLM to generate the answer

================================================================================
"""

# ================================================================================
# SECTION 1: IMPORTS - Bring in all the tools we need
# ================================================================================

import os
import warnings
import streamlit as st
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Suppress warning messages for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import LangChain tools for document processing
from langchain_community.document_loaders import (
    PyPDFLoader,              # Read PDF files
    CSVLoader,                # Read CSV files
    TextLoader,               # Read TXT files
    UnstructuredWordDocumentLoader,    # Read Word files
    UnstructuredExcelLoader,  # Read Excel files
    UnstructuredHTMLLoader,   # Read HTML files
)

# Import LangChain tools for chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import LangChain tools for vector storage and embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import LangChain tools for LLM and prompting
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core import runnables


# ================================================================================
# SECTION 2: CONFIGURATION - Settings for the RAG system
# ================================================================================

# Load environment variables from .env file
load_dotenv()

# What AI models to use?
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast & lightweight (384-dim)
CHAT_MODEL = "llama-3.1-8b-instant"  # Groq's fastest model

# RAG Settings
CHUNK_SIZE = 1000          # How many characters per chunk?
CHUNK_OVERLAP = 200        # How much overlap between chunks?
RETRIEVAL_K = 3            # How many chunks to use for answering?

# Where to store the database?
CHROMA_DB_PATH = "./.chroma_db"  # Local folder in your project

# What file types are supported?
SUPPORTED_FORMATS = {
    "pdf": "PDF Documents",
    "csv": "CSV Spreadsheets",
    "txt": "Text Files",
    "md": "Markdown Files",
    "docx": "Word Documents (.docx)",
    "xlsx": "Excel Spreadsheets",
    "html": "HTML Files",
}


# ================================================================================
# SECTION 3: HELPER FUNCTIONS - Useful tools for file handling
# ================================================================================

def get_groq_api_key() -> Optional[str]:
    """
    Get the Groq API key from two possible locations.
    
    First tries: .streamlit/secrets.toml (best for local testing)
    Then tries: GROQ_API_KEY environment variable (best for production)
    
    Returns:
        str: The API key if found
        None: If API key is missing
    """
    # Try to get from Streamlit secrets first
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    
    # Fallback to environment variable
    return os.getenv("GROQ_API_KEY")


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save a file that user uploaded to our computer temporarily.
    
    Why? LangChain loaders need actual file paths, not just the file content.
    
    Args:
        uploaded_file: The file the user uploaded through Streamlit
        
    Returns:
        Path: The location where we saved the file
    """
    # Create temp_uploads folder if it doesn't exist
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    # Save the file
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def get_file_extension(filename: str) -> str:
    """Get the file extension (like 'pdf', 'csv') from a filename."""
    return Path(filename).suffix.lower().lstrip(".")


# ================================================================================
# SECTION 4: DOCUMENT LOADING - Load different document types
# ================================================================================

def load_document(file_path: Path) -> List:
    """
    Load any supported document type and convert to LangChain documents.
    
    This function is like a dispatcher that routes files to the right loader.
    - PDF file? → Use PDFLoader
    - CSV file? → Use CSVLoader
    - Etc.
    
    What does "Load" mean?
    - Read the file
    - Extract text
    - Return as LangChain Document objects (which have .page_content and .metadata)
    
    Args:
        file_path: Where the file is located on disk
        
    Returns:
        List: List of LangChain Document objects
    """
    # Get file extension (pdf, csv, txt, etc.)
    extension = get_file_extension(file_path.name)
    
    try:
        if extension == "pdf":
            # PDFLoader reads PDF files page by page
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        
        elif extension == "csv":
            # CSVLoader reads CSV files and treats them as documents
            loader = CSVLoader(str(file_path))
            documents = loader.load()
        
        elif extension in ["txt", "md"]:
            # TextLoader reads plain text and markdown files
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
        
        elif extension == "docx":
            # UnstructuredWordDocumentLoader reads Word documents
            loader = UnstructuredWordDocumentLoader(str(file_path))
            documents = loader.load()
        
        elif extension == "xlsx":
            # UnstructuredExcelLoader reads Excel spreadsheets
            loader = UnstructuredExcelLoader(str(file_path))
            documents = loader.load()
        
        elif extension == "html":
            # UnstructuredHTMLLoader reads HTML files
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
        
        else:
            # If file type is not supported, show error
            raise ValueError(f"Unsupported file format: {extension}")
        
        return documents
    
    except Exception as e:
        raise Exception(f"Error loading {extension.upper()}: {str(e)}")


# ================================================================================
# SECTION 5: TEXT SPLITTING - Break documents into chunks
# ================================================================================

def split_documents(documents: List, chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> List:
    """
    Split documents into smaller chunks for embedding.
    
    Why split?
    - Large documents are too big to embed properly
    - Smaller chunks are more relevant for retrieval
    - We can retrieve multiple chunks for one question
    
    How does RecursiveCharacterTextSplitter work?
    - Tries to split on paragraph boundaries first (\n\n)
    - If chunks are still too big, splits on sentence boundaries (\n)
    - If still too big, splits on spaces (words)
    - Last resort: splits individual characters
    
    This keeps related text together! Smart.
    
    Args:
        documents: List of documents from the loaders
        chunk_size: How many characters per chunk (default 1000)
        chunk_overlap: How much should chunks overlap (default 200)
        
    Returns:
        List: List of smaller document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # Try these in order
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


# ================================================================================
# SECTION 6: EMBEDDINGS & VECTOR STORAGE - Convert text to vectors
# ================================================================================

@st.cache_resource
def load_embeddings_model():
    """
    Load the embeddings model ONCE and cache it.
    
    What's caching?
    - Without @st.cache_resource: Model reloads every time you interact
    - With @st.cache_resource: Model loads once, reused forever
    - Makes the app much faster!
    
    What are embeddings?
    - Text converted to vectors (lists of numbers)
    - Semantically similar text has similar vectors
    - We use cosine similarity to find relevant chunks
    
    What model?
    - all-MiniLM-L6-v2 from HuggingFace
    - Lightweight (only 22MB)
    - Fast (can embed 1000s of chunks quickly)
    - Good quality (trained on semantic similarity)
    
    Returns:
        HuggingFaceEmbeddings: The embeddings model, ready to use
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use CPU (macOS compatible)
        encode_kwargs={"normalize_embeddings": True},  # Normalize for cosine similarity
    )


def create_or_load_vectorstore(documents: Optional[List] = None) -> Chroma:
    """
    Create a new vector database OR load an existing one.
    
    What's a vector database?
    - Stores embeddings (vectors) for fast similarity search
    - Chroma is lightweight and works locally
    - Data persists in ./.chroma_db folder
    
    If documents provided:
    - Create a NEW vector store
    - Convert documents to embeddings
    - Save to disk
    
    If no documents:
    - Load EXISTING vector store from disk
    - Useful when app restarts
    
    Args:
        documents: If provided, create new vectorstore. If None, load existing.
        
    Returns:
        Chroma: The vector store, ready for similarity search
    """
    # Load the embeddings model
    embeddings = load_embeddings_model()
    
    if documents:
        # Creating a NEW vector store from documents
        print(f"Creating new vector store with {len(documents)} documents...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name="rag_documents",
        )
    else:
        # Loading an EXISTING vector store from disk
        print(f"Loading existing vector store from {CHROMA_DB_PATH}...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name="rag_documents",
        )
    
    return vectorstore


# ================================================================================
# SECTION 7: RETRIEVAL - Find relevant chunks for a question
# ================================================================================

def retrieve_context(query: str, vectorstore: Chroma, k: int = RETRIEVAL_K) -> tuple[List, str]:
    """
    Find the k most relevant document chunks for the user's question.
    
    How does this work?
    1. Convert the question to an embedding (vector)
    2. Compare it against all chunk embeddings using cosine similarity
    3. Return the top k most similar chunks
    
    Why?
    - We're finding context that's relevant to the question
    - This context will be given to the LLM to generate the answer
    - More relevant context = better answers
    
    Args:
        query: The user's question (text)
        vectorstore: The Chroma vector store
        k: How many chunks to retrieve (default 3)
        
    Returns:
        tuple: 
            - List of LangChain Document objects (the chunks)
            - String of formatted context (all chunks joined together)
    """
    # Use Chroma's similarity_search to find relevant documents
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    
    # Join all the chunks into one context string with separators
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    return retrieved_docs, context


# ================================================================================
# SECTION 8: PROMPT & GENERATION - Use LLM to answer questions
# ================================================================================

@st.cache_resource
def create_rag_chain(groq_api_key: str):
    """
    Create the RAG pipeline that answers questions.
    
    What's a "RAG chain"?
    - A pipeline that combines Retrieval + Generation
    - Retrieval: Find relevant context from documents
    - Generation: Use LLM to generate answer based on context
    
    What does it do?
    1. Takes a question as input
    2. Retrieves relevant document chunks from Chroma
    3. Formats a prompt: "Here's context: ... Now answer this: ..."
    4. Sends prompt to Groq LLM
    5. LLM returns the answer
    6. Returns answer to user
    
    Args:
        groq_api_key: Your Groq API key for authentication
        
    Returns:
        tuple:
            - rag_chain: The runnable chain for answering questions
            - vectorstore: The vector store (for showing sources)
    """
    # Load the vector store with our documents
    vectorstore = create_or_load_vectorstore()
    
    # Create a retriever from the vector store
    # This is what will retrieve relevant chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )
    
    # Define the prompt template
    # This is the structure of the prompt we'll send to Groq
    prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Your job is to answer questions based on the documents provided.

IMPORTANT RULES:
1. Answer ONLY from the context provided below
2. If the context doesn't have relevant information, say: "I couldn't find relevant information in the documents."
3. Be clear and thorough - provide complete answers with all relevant details
4. Quote from the documents when it helps to support your answer
5. Use proper formatting (paragraphs, bullet points, lists) for readability when appropriate

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{question}

YOUR ANSWER:""")
    
    # Create the Groq LLM instance
    # This is what actually generates the answers
    llm = ChatGroq(
        model=CHAT_MODEL,
        groq_api_key=groq_api_key,
        temperature=0.3,  # Lower = more consistent, Higher = more creative
    )
    
    # Build the RAG chain using LangChain
    # Think of | as a pipe: left side output goes to right side input
    # This creates a pipeline:
    # 1. Pass question through
    # 2. Retrieve relevant docs
    # 3. Format with prompt
    # 4. Send to LLM
    # 5. Parse output
    rag_chain = (
        {
            "context": retriever,  # Get context from retriever
            "question": RunnablePassthrough(),  # Pass through the question unchanged
        }
        | prompt_template  # Format into our prompt structure
        | llm  # Send to Groq LLM
        | StrOutputParser()  # Convert output to string
    )
    
    return rag_chain, vectorstore


# ================================================================================
# SECTION 9: STREAMLIT UI - The user interface
# ================================================================================

def main():
    """
    The main Streamlit application.
    
    This controls everything the user sees and interacts with.
    """
    
    # ========================================================================
    # Page Configuration
    # ========================================================================
    st.set_page_config(
        page_title="RAG Chroma LangChain",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # ========================================================================
    # Session State - Remember things between interactions
    # ========================================================================
    # Session state is how Streamlit apps remember variables when they re-run
    # Every time the user clicks something, Streamlit re-runs the entire script
    # Session state keeps variables alive between runs
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None  # The RAG pipeline
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None  # The vector database
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Previous questions & answers
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False  # Are documents ready?
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0  # How many documents?
    
    # ========================================================================
    # Main Page Title & Description
    # ========================================================================
    st.title("📚 RAG with Chroma & LangChain")
    st.markdown("""
    ### Ask questions about your documents using AI
    
    This app combines three powerful tools:
    - **Chroma**: Fast vector database for similarity search
    - **LangChain**: Framework for building AI applications
    - **Groq**: Fast LLM for generating answers
    """)
    
    # ========================================================================
    # SIDEBAR: Configuration & Document Upload
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Setup & Configuration")
        
        # Check API Key
        groq_api_key = get_groq_api_key()
        if groq_api_key:
            st.success("✅ Groq API Key loaded")
        else:
            st.error("❌ Groq API Key not found")
            st.markdown("""
            **To setup your API key:**
            
            1. **Option A - Local Testing:**
               Create `.streamlit/secrets.toml`:
               ```toml
               GROQ_API_KEY = "gsk_xxxxx"
               ```
            
            2. **Option B - Environment Variable:**
               ```bash
               export GROQ_API_KEY="gsk_xxxxx"
               ```
            
            3. **Option C - Streamlit Cloud:**
               Go to Settings → Secrets and add the key
            
            Get free API key: https://console.groq.com
            """)
            return
        
        st.divider()
        
        # Document Upload Section
        st.header("📄 Upload Documents")
        
        # Show what formats are supported
        with st.expander("ℹ️ Supported File Formats"):
            for ext, name in SUPPORTED_FORMATS.items():
                st.write(f"• **{ext.upper()}** - {name}")
        
        # File upload widget
        uploaded_files = st.file_uploader(
            "Select files to upload",
            type=list(SUPPORTED_FORMATS.keys()),
            accept_multiple_files=True,
            help="Upload one or more documents. They will be processed and indexed."
        )
        
        # Advanced Settings (collapsible)
        with st.expander("⚙️ Advanced Settings"):
            chunk_size = st.slider(
                "Chunk size (characters)",
                min_value=200,
                max_value=2000,
                value=CHUNK_SIZE,
                step=100,
                help="Smaller = more specific, Larger = more context"
            )
            chunk_overlap = st.slider(
                "Chunk overlap (characters)",
                min_value=0,
                max_value=500,
                value=CHUNK_OVERLAP,
                step=50,
                help="Overlap helps maintain context between chunks"
            )
            retrieval_k = st.slider(
                "Chunks to retrieve",
                min_value=1,
                max_value=10,
                value=RETRIEVAL_K,
                help="How many document chunks to use for answering?"
            )
        
        # Process Button
        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("❌ Please upload at least one document")
            else:
                with st.spinner("🔄 Processing your documents..."):
                    try:
                        # Step 1: Load all documents
                        st.info("📖 Step 1: Loading documents...")
                        all_documents = []
                        progress_bar = st.progress(0)
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            # Save the uploaded file temporarily
                            file_path = save_uploaded_file(uploaded_file)
                            
                            # Load using LangChain loaders
                            documents = load_document(file_path)
                            all_documents.extend(documents)
                            
                            # Update progress
                            progress = (i + 1) / len(uploaded_files)
                            progress_bar.progress(progress)
                        
                        st.success(f"✅ Loaded {len(all_documents)} documents")
                        
                        # Step 2: Split documents into chunks
                        st.info("✂️ Step 2: Splitting documents into chunks...")
                        chunks = split_documents(
                            all_documents,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        st.success(f"✅ Created {len(chunks)} chunks")
                        
                        # Step 3: Create vector store & RAG chain
                        st.info("🔗 Step 3: Creating vector store...")
                        vectorstore = create_or_load_vectorstore(documents=chunks)
                        rag_chain, _ = create_rag_chain(groq_api_key)
                        st.success("✅ Vector store created!")
                        
                        # Step 4: Save to session state
                        st.info("💾 Step 4: Saving to session...")
                        st.session_state.rag_chain = rag_chain
                        st.session_state.vectorstore = vectorstore
                        st.session_state.documents_loaded = True
                        st.session_state.doc_count = len(all_documents)
                        st.session_state.retrieval_k = retrieval_k
                        st.success("✅ RAG system ready! Start asking questions →")
                        
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Clear Data Button (only show if documents are loaded)
        if st.session_state.documents_loaded:
            st.divider()
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state.rag_chain = None
                st.session_state.vectorstore = None
                st.session_state.chat_history = []
                st.session_state.documents_loaded = False
                st.session_state.doc_count = 0
                st.rerun()
        
        # Status Section
        st.divider()
        st.header("📊 Status")
        if st.session_state.documents_loaded:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", st.session_state.doc_count)
            with col2:
                st.metric("Messages", len(st.session_state.chat_history))
        else:
            st.info("💡 Upload documents to get started")
    
    # ========================================================================
    # MAIN AREA: Chat Interface
    # ========================================================================
    
    # Show instructions if no documents loaded
    if not st.session_state.documents_loaded:
        st.info("👈 **Upload documents in the sidebar** to get started!")
        
        # Show how RAG works
        with st.expander("📖 How RAG Works (Beginner Explanation)", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                ### 1️⃣ LOAD
                
                Upload your documents:
                - PDF
                - Word
                - Excel
                - CSV
                - Text
                - HTML
                """)
            
            with col2:
                st.markdown("""
                ### 2️⃣ PROCESS
                
                The app prepares documents:
                - Splits into chunks
                - Converts to embeddings
                - Stores in vector DB
                
                This takes 5-10 seconds
                """)
            
            with col3:
                st.markdown("""
                ### 3️⃣ ANSWER
                
                When you ask a question:
                - Finds relevant chunks
                - Sends to Groq LLM
                - Returns answer + sources
                
                Takes 1-2 seconds
                """)
        
        # Show example
        st.divider()
        st.subheader("💡 Example Usage")
        st.markdown("""
        1. Upload a PDF of your company handbook
        2. Click "Process & Index Documents"
        3. Ask questions like:
           - "What's the PTO policy?"
           - "How do I request time off?"
           - "What benefits do we offer?"
        4. Get instant answers with source citations
        """)
        
        return
    
    # Chat history display
    st.subheader("💬 Conversation")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input box
    if query := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):
                try:
                    # Use RAG chain to answer
                    answer = st.session_state.rag_chain.invoke(query)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add to history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    
                    # Show sources
                    with st.expander("📎 Show Sources"):
                        k = st.session_state.get("retrieval_k", RETRIEVAL_K)
                        retrieved_docs, _ = retrieve_context(
                            query,
                            st.session_state.vectorstore,
                            k=k,
                        )
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"**Source {i}:**")
                            # Show first 300 characters
                            preview = doc.page_content[:300]
                            if len(doc.page_content) > 300:
                                preview += "..."
                            st.markdown(f"> {preview}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# ================================================================================
# SECTION 10: APPLICATION ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    main()
