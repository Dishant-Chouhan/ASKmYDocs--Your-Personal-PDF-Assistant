# create_memory.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import streamlit as st

load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_memory_from_pdf(pdf_paths, append=False):
    """
    Create or append to a FAISS vector store from PDF files.
    Args:
        pdf_paths: List of paths to PDF files.
        append: If True, append to existing vector store; if False, create new.
    Returns:
        FAISS vector store and list of document snippets.
    """
    try:
        documents = []
        snippets = []
        progress = st.progress(0)
        for i, path in enumerate(pdf_paths):
            try:
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                if loaded_docs:
                    snippets.append(f"{os.path.basename(path)}: {loaded_docs[0].page_content[:100]}...")
            except Exception as e:
                st.warning(f"⚠️ Failed to process {os.path.basename(path)}: {str(e)}")
            progress.progress((i + 1) / len(pdf_paths))

        if not documents:
            raise ValueError("No valid documents were loaded from the provided PDFs.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        if append and os.path.exists(DB_FAISS_PATH):
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            db.add_documents(text_chunks)
        else:
            db = FAISS.from_documents(text_chunks, embedding_model)

        db.save_local(DB_FAISS_PATH)
        return db, snippets
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None, []

def list_documents_in_vectorstore():
    """
    List unique document filenames in the FAISS vector store.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return list({os.path.basename(doc.metadata.get("source", "Unknown")) for doc in db.docstore._dict.values()})
    except Exception as e:
        st.error(f"❌ Error listing documents: {str(e)}")
        return []

def delete_documents_by_filename(filenames):
    """
    Delete documents from the FAISS vector store by filename.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

        keys_to_delete = [
            doc_id for doc_id, doc in db.docstore._dict.items()
            if os.path.basename(doc.metadata.get("source", "")) in filenames
        ]
        for key in keys_to_delete:
            db.docstore._dict.pop(key, None)

        db.save_local(DB_FAISS_PATH)
        return True
    except Exception as e:
        st.error(f"❌ Error deleting documents: {str(e)}")
        return False