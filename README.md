# 📄 ASKmYDocs – Your Personal PDF Assistant

ASKmYDocs is a user-friendly PDF-based Q&A chatbot built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), and Hugging Face models. Upload one or more PDF documents and ask questions about their content — your assistant will fetch relevant answers with source references, highlighted keywords, and dynamic follow-up suggestions!

---

## 🔍 Features

- 💬 **Chat-based Q&A**: Ask questions about your PDFs and get concise, context-aware answers.
- 📂 **Multi-PDF Upload**: Upload multiple PDFs at once to build a knowledge base.
- 🧠 **RAG-based Memory**: Uses Retrieval-Augmented Generation (RAG) with FAISS for efficient vector similarity search.
- 📑 **Source Attribution**: Displays source documents with file names, page numbers, and highlighted keyword excerpts in a table.
- 🔄 **Interactive Controls**:
  - Regenerate answers for alternative responses.
  - Download chat history as a text file.
  - Delete specific documents or reset the entire vector store.
  - Clear or selectively delete chat history messages.
- 🧾 **Document Management**: Refresh document list, view knowledge base stats (documents and chunks), and delete selected PDFs.
- 💡 **Dynamic Follow-Up Questions**: Clickable suggestions based on your queries to guide exploration.
- 📜 **Document Previews**: Preview snippets of uploaded PDFs.
- 🔍 **Chat History Navigation**: Filter and scroll through past messages in the sidebar.
- 📊 **Session Insights**: Displays the number of questions asked in the current session.

---

## 🛠️ Tech Stack

| Tool/Library | Role |
|--------------|------|
| 🐍 **Python** | Core programming language |
| 🧠 **[LangChain](https://www.langchain.com/)** | RAG architecture, chaining logic, and prompt management |
| 🧠 **[FAISS](https://github.com/facebookresearch/faiss)** | Vector similarity search for document retrieval |
| 💬 **[Streamlit](https://streamlit.io/)** | Web-based UI and interactivity |
| 🤗 **Hugging Face Hub** | LLMs and embedding models for question answering and text embedding |
| 🧾 **PyPDFLoader** | PDF loading and text extraction |
| 🔐 **python-dotenv** | Environment variable management for secure API key handling |

---

## 🤖 Models Used

### 1. **[mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)**
- **Type**: Instruction-tuned Large Language Model (LLM)
- **Role**: Generates answers to user queries based on retrieved document context
- **Strengths**: Lightweight 7B parameter model optimized for natural language instructions
- **Hosted**: Via `HuggingFaceEndpoint` (requires Hugging Face API key)

### 2. **[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**
- **Type**: Embedding model
- **Role**: Converts PDF text chunks into vector representations for similarity search
- **Strengths**: Fast, lightweight, and accurate for semantic similarity tasks

---
## 📚 Usage Guide

### 📄 Upload PDFs

- Use the file uploader on the main interface to upload one or more PDF files.
- Check **"Append new PDFs to existing knowledge base"** to add files without overwriting current data.
- View document previews and knowledge base stats (e.g., number of documents and text chunks) in the sidebar.

---

### 🗂 Manage Documents (Sidebar)

- **Refresh Document List**: Update the list of uploaded PDFs.
- **Delete Selected PDFs**: Remove specific documents from the knowledge base.
- **Reset Vector Store**: Clear all data in `vectorstore/db_faiss` to start fresh.

---

### 💬 Ask Questions

- Enter your question in the chat input at the bottom of the main interface.
- Get answers with highlighted source excerpts (including file names and page numbers).
- Click suggested follow-up questions to explore related topics.

---

### 📖 Manage Chat History (Sidebar)

- **Clear Chat History**: Delete all messages from the current session.
- **Delete Selected Messages**: Remove specific messages from chat history.
- **Navigate History**: Filter messages by **All**, **User**, or **Assistant** and scroll to view past interactions.

---

### ⚙️ Additional Actions

- **Copy Answer**: Copy the latest response to your clipboard.
- **Regenerate Answer**: Request a new response for the same question.
- **Download Chat History**: Save the current session as a text file.
