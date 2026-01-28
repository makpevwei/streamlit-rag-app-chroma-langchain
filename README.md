# 📚 RAG with Chroma & LangChain - Beginner's Guide

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Groq API Key
Create or edit `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_xxxxx"
```

Get a free API key: [console.groq.com](https://console.groq.com)

### 3. Run the App
```bash
streamlit run rag_chroma_langchain.py
```

Open browser to: `http://localhost:8501`

### 4. Test It
- Upload a PDF or document
- Click "Process & Index Documents"
- Ask a question about your document
- Get answer with sources!

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Normally, ChatGPT answers based on what it learned during training. RAG lets you answer based on YOUR documents:

```
You ask:   "What's in my document?"
          ↓
RAG finds: Relevant chunks from YOUR documents
          ↓
LLM says:  "Based on your document, the answer is..."
```

### The RAG Pipeline (7 Steps)

```
1. LOADING    📄 Load your documents
              ↓
2. SPLITTING  ✂️ Break into chunks
              ↓
3. EMBEDDING  🔢 Convert to vectors
              ↓
4. STORAGE    💾 Save in Chroma DB
              ↓
5. RETRIEVAL  🔍 Find relevant chunks
              ↓
6. PROMPTING  ✍️ Build context prompt
              ↓
7. GENERATION 🤖 LLM generates answer
```

---

## Features

✅ **Upload Multiple Formats**
- PDF documents
- Word files (.docx)
- Excel spreadsheets (.xlsx)
- CSV files
- Text files (.txt)
- Markdown files (.md)
- HTML files (.html)

✅ **Smart Chunking**
- Automatically splits documents into manageable pieces
- Preserves semantic meaning
- Customizable chunk size and overlap

✅ **Fast Retrieval**
- Uses Chroma vector database
- Finds relevant chunks in milliseconds
- Shows sources with answers

✅ **AI-Powered Answers**
- Uses Groq's fast LLM (llama-3.1-8b-instant)
- Accurate and contextual responses
- Sources shown for verification

---

## Understanding the Code

### For Complete Beginners

The app is organized into **10 sections**:

| Section | What It Does | For Beginners |
|---------|---------|---------|
| 1 - Imports | Load all the tools | "Getting the toolkit" |
| 2 - Config | Set up settings | "Configuring the tools" |
| 3 - Helpers | Utility functions | "Helper functions" |
| 4 - Loading | Read documents | "Read PDF, Word, CSV" |
| 5 - Splitting | Break into chunks | "Split big texts into small pieces" |
| 6 - Embeddings | Convert to vectors | "Turn text into numbers AI understands" |
| 7 - Retrieval | Find relevant chunks | "Search for relevant parts" |
| 8 - Generation | Answer with LLM | "Use AI to generate answers" |
| 9 - Streamlit UI | The interface | "The buttons and boxes you click" |
| 10 - Entry | Start the app | "Where the app starts" |

### Key Concepts Explained

**Embeddings**
- Text converted to vectors (lists of numbers)
- Similar text has similar vectors
- Used for finding relevant chunks

**Vector Database (Chroma)**
- Stores embeddings for fast search
- Finds similar chunks using cosine similarity
- Much faster than searching raw text

**LangChain**
- Framework for building AI applications
- Connects loaders → splitters → embeddings → LLM
- Makes complex tasks simple

**Groq API**
- Fast inference LLM API
- llama-3.1-8b-instant model
- Returns answers quickly (1-2 seconds)

---

## How to Use

### Basic Usage

1. **Click "Upload Documents"**
   - Select one or more files
   - Supported: PDF, Word, Excel, CSV, Text, HTML

2. **Click "Process & Index Documents"**
   - This runs through the 7-step pipeline
   - Takes 5-10 seconds depending on file size
   - Creates searchable index

3. **Ask a Question**
   - Type in the chat box at the bottom
   - Press Enter
   - App finds relevant chunks and generates answer

4. **View Sources**
   - Click "Show Sources" below answer
   - See which document chunks were used
   - Verify answer is based on your documents

### Advanced Settings

**Chunk Size**
- Smaller (200-500): More specific answers, less context
- Larger (1500-2000): More context, less specific
- Default (1000): Good balance

**Chunk Overlap**
- Prevents losing information at chunk boundaries
- 200 characters is usually good
- Adjust if answers seem disconnected

**Chunks to Retrieve (k)**
- How many chunks to use for answering?
- 1-3: Specific answers
- 4-5: More context
- 5+: Full context but slower

---

## Troubleshooting

### "API Key not found"
- Add to `.streamlit/secrets.toml`:
  ```toml
  GROQ_API_KEY = "your_key_here"
  ```
- Or set environment variable:
  ```bash
  export GROQ_API_KEY="your_key_here"
  ```

### "Module not found" error
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### "No relevant information found"
- Your chunks might not contain the answer
- Try adjusting chunk size (make it larger)
- Upload more documents
- Ask more specific questions

### App runs slowly
- Reduce number of retrieval chunks (k)
- Reduce chunk size
- Clear and re-index documents

### "Error loading PDF"
- Try opening PDF in Adobe Reader first
- Some PDFs are scanned images, not text
- Convert scanned PDFs to text first

---

## Deployment

### Local Testing
```bash
streamlit run rag_chroma_langchain.py
```

### Streamlit Cloud
1. Push to GitHub (remember to .gitignore .env!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create app from GitHub
4. Add API key in Settings > Secrets
5. Deploy!

### Docker
```bash
docker build -t rag-chroma .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key rag-chroma
```

### Manual Server
```bash
git clone your-repo
cd rag_chroma_langchain
pip install -r requirements.txt
export GROQ_API_KEY="your_key"
streamlit run rag_chroma_langchain.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         User Interface (Streamlit)              │
│  - Upload files                                  │
│  - Chat interface                                │
│  - Show sources                                  │
└────────────┬────────────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
   ┌──▼──┐      ┌──▼──────┐
   │Load │      │Split &  │
   │Docs │      │Embed    │
   └──┬──┘      └──┬──────┘
      │            │
      └────┬───────┘
           │
      ┌────▼──────────┐
      │ Chroma Vector │
      │  Database     │
      └────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼──────────┐
│Retrieve│   │Generate with │
│Chunks  │   │Groq LLM      │
└────┬───┘   └───┬──────────┘
     │           │
     └─────┬─────┘
           │
      ┌────▼──────┐
      │ Answer    │
      │+ Sources  │
      └───────────┘
```

---

## Technologies Used

| Tool | Purpose | Why |
|------|---------|-----|
| Streamlit | Web UI | Easy to build, fast to develop |
| LangChain | AI framework | Simplifies complex pipelines |
| Chroma | Vector DB | Lightweight, local persistence |
| HuggingFace | Embeddings | Free, high quality, fast |
| Groq | LLM | Fastest inference available |
| Python | Language | Versatile, good libraries |

---

## Performance Tips

- **First Load**: 10-15 seconds (model download)
- **Processing**: 5-10 seconds per document
- **Retrieval**: 1-2 seconds per query
- **Memory**: 1-2 GB typical

To make it faster:
- Use smaller documents
- Reduce chunk size
- Use fewer retrieval chunks (k=1-2)
- Close other apps

---

## Common Questions

**Q: Can I use my own LLM instead of Groq?**
A: Yes! Replace ChatGroq with any LangChain LLM (OpenAI, Anthropic, etc.)

**Q: Can I use different embeddings?**
A: Yes! Any HuggingFace model works. Swap model name in config.

**Q: Can I save chat history?**
A: Yes! The app stores in session_state. Add database saving in `main()`

**Q: How many documents can I upload?**
A: Technically unlimited, but each document loads into memory.

**Q: Is my data private?**
A: Yes! Chroma stores locally. Embeddings don't leave your machine.

---

## Learning Path for Beginners

1. **Understand RAG**
   - Read the "What is RAG?" section above

2. **Run the App**
   - Follow Quick Start
   - Try with sample documents

3. **Explore the Code**
   - Read Section 1-3 (Imports, Config, Helpers)
   - Understanding what each part does

4. **Customize It**
   - Change CHUNK_SIZE in config
   - Adjust prompts in Section 8
   - Add your own document types

5. **Deploy It**
   - Push to GitHub
   - Deploy to Streamlit Cloud
   - Share with others

---

## Resources

**Groq API**: https://console.groq.com  
**LangChain Docs**: https://python.langchain.com  
**Chroma Docs**: https://docs.trychroma.com  
**Streamlit Docs**: https://docs.streamlit.io  

---

## Support

- Check `.streamlit/secrets.toml` has API key
- Check `.gitignore` includes sensitive files
- Read code comments (each section is documented)
- Refer to "Troubleshooting" section above

---

**Created**: January 2026  
**Status**: Production Ready ✅  
**Level**: Beginner Friendly  
**Last Updated**: Today  

