# d.py

import os
import streamlit as st
import tempfile
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import shutil
import re

from create_memory import create_memory_from_pdf, list_documents_in_vectorstore, delete_documents_by_filename

load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ------------------------- Helper Functions -------------------------

def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def cleanup_temp_files(pdf_paths):
    for path in pdf_paths:
        try:
            os.remove(path)
            os.rmdir(os.path.dirname(path))
        except Exception as e:
            st.warning(f"⚠️ Could not clean up {path}: {str(e)}")

def highlight_keywords(text, query):
    """Highlight query keywords in text using markdown bold."""
    words = re.findall(r'\w+', query.lower())
    for word in words:
        text = re.sub(rf'\b({word})\b', r'**\1**', text, flags=re.IGNORECASE)
    return text

def generate_follow_up_questions(query):
    """Generate dynamic follow-up questions based on the query."""
    words = re.findall(r'\w+', query.lower())
    main_topic = words[0] if words else "this topic"
    return [
        f"Can you provide more details on {main_topic}?",
        f"What else does the document say about {main_topic}?",
        f"Can you summarize the main points related to {main_topic}?"
    ]

def set_custom_prompt():
    custom_prompt_template = """
You are ASKmYDocs, a friendly and intelligent assistant designed to provide clear, accurate, and concise answers based on the content of uploaded PDF documents.

Rules:
- Use only the provided context to answer the question.
- Consider the chat history to maintain context for follow-up questions.
- Provide a concise answer (1-3 sentences) unless more detail is requested.
- If the question is unclear, ask for clarification with a suggested rephrasing.
- If the answer is not in the context, respond with "I don't know based on the provided documents" and suggest rephrasing.
- Reference the source document (e.g., PDF file name) when relevant.
- Use a professional yet conversational tone with markdown formatting.

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer:
"""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "history", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=os.environ.get("HF_TOKEN")
    )

def get_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )

@st.cache_resource
def load_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        if os.path.exists(DB_FAISS_PATH):
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            return db
        else:
            st.warning("⚠️ No vector store found. Please upload PDFs to create one.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        return None

# ------------------------- Streamlit UI -------------------------

st.set_page_config(page_title="ASKmYDocs", layout="wide")
st.title("📄 ASKmYDocs - Your Personal PDF Assistant")

# Sidebar for document and chat management
with st.sidebar:
    st.header("📂 Document Manager")
    if st.button("🔄 Refresh Document List"):
        with st.spinner("🔄 Refreshing document list..."):
            st.session_state.available_docs = list_documents_in_vectorstore()
            st.success(f"✅ Document list refreshed! Found {len(st.session_state.available_docs)} documents.")

    if "available_docs" not in st.session_state:
        st.session_state.available_docs = list_documents_in_vectorstore()

    # Display document count and chunk stats
    if os.path.exists(DB_FAISS_PATH):
        vectorstore_temp = load_vectorstore()
        if vectorstore_temp:
            chunk_count = len(vectorstore_temp.docstore._dict)
            st.info(f"📚 Knowledge base: {len(st.session_state.available_docs)} documents, {chunk_count} chunks")

    selected_files = st.multiselect("🧾 Delete Selected PDFs", st.session_state.available_docs, help="Select documents to delete from the knowledge base.")
    if st.button("❌ Delete Selected Documents"):
        if selected_files:
            with st.spinner("🗑️ Deleting documents..."):
                success = delete_documents_by_filename(selected_files)
                if success:
                    st.session_state.available_docs = list_documents_in_vectorstore()
                    st.success("✅ Selected documents deleted!")
                else:
                    st.error("❌ Failed to delete documents.")
        else:
                
            st.warning("⚠️ Please select at least one document.")

    if st.button("🔄 Reset Vector Store"):
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)
            st.session_state.available_docs = []
            st.success("✅ Vector store reset successfully! All document data has been cleared.")
        else:
            st.info("ℹ️ No vector store to reset.")

    st.header("💬 Chat Management")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.success("✅ Chat history cleared!")

    # Chat history navigation
    with st.expander("💬 Chat History", expanded=False):
        filter_option = st.selectbox("Filter Messages", ["All", "User", "Assistant"])
        selected_messages = st.multiselect("Select Messages to Delete", [f"{idx}: {msg['role'].capitalize()}: {msg['content'][:50]}..." for idx, msg in enumerate(st.session_state.get('messages', []))], key="delete_messages")
        if st.button("🗑️ Delete Selected Messages"):
            if selected_messages:
                indices = [int(msg.split(":")[0]) for msg in selected_messages]
                st.session_state.messages = [msg for idx, msg in enumerate(st.session_state.messages) if idx not in indices]
                st.success("✅ Selected messages deleted!")
            else:
                st.warning("⚠️ Please select at least one message.")

        for idx, message in enumerate(st.session_state.get('messages', [])):
            if filter_option == "All" or message['role'] == filter_option.lower():
                st.chat_message(message['role']).markdown(message['content'])
                if st.button("🔝 Scroll to Message", key=f"scroll_{idx}"):
                    st.markdown(f"<script>window.scrollTo(0, document.getElementById('chat_{idx}').offsetTop);</script>", unsafe_allow_html=True)

# Main UI
append_to_existing = st.checkbox("Append new PDFs to existing knowledge base", value=False, help="Check to add new PDFs without overwriting existing data.")

uploaded_files = st.file_uploader("📤 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True, help="Upload PDFs to query their content.")

# File Upload & Processing
if uploaded_files:
    with st.spinner("🔄 Processing PDFs..."):
        pdf_paths = [save_uploaded_file(file) for file in uploaded_files]
        vectorstore, snippets = create_memory_from_pdf(pdf_paths, append=append_to_existing)
        if vectorstore:
            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s).")
            with st.expander("📜 Document Previews"):
                for snippet in snippets:
                    st.markdown(f"- {snippet}")
            cleanup_temp_files(pdf_paths)
            st.session_state.available_docs = list_documents_in_vectorstore()
        else:
            st.error("❌ Failed to process the PDFs.")
else:
    vectorstore = load_vectorstore()

# Chat Interface
if vectorstore:
    st.info("✅ Ready to answer questions based on your documents!")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Chat history summary
    question_count = sum(1 for msg in st.session_state.messages if msg['role'] == 'user')
    st.caption(f"💬 {question_count} question(s) asked in this session.")

    # Display current chat
    for idx, message in enumerate(st.session_state.messages):
        st.chat_message(message['role']).markdown(message['content'], unsafe_allow_html=True)

    # Add help text above chat input
    st.caption("Type your question about the PDFs here.")
    prompt = st.chat_input("💬 Ask your question")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            with st.spinner("🤖 ASKmYDocs is thinking..."):
                time.sleep(1)  # Simulate processing delay
                qa_chain = get_qa_chain(vectorstore)
                history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-3:]])

                # Pass all required inputs explicitly
                response = qa_chain({"query": prompt, "history": history})
                result = response["result"].strip()
                sources = response["source_documents"]

                # Check for "I don't know" response and suggest rephrasing
                if "I don't know" in result:
                    final_response = f"**Answer:** {result}\n\n**Tip:** Try rephrasing your question, e.g., 'Can you find specific details about [topic] in the document?'"
                else:
                    final_response = f"**Answer:** {result}"

                st.chat_message('assistant').markdown(final_response)
                st.session_state.messages.append({'role': 'assistant', 'content': final_response})

                # Source documents in a table
                if sources:
                    st.markdown("**Source Documents:**")
                    source_data = [
                        {
                            "File": os.path.basename(doc.metadata.get('source', 'Unknown')),
                            "Page": doc.metadata.get('page', 'N/A'),
                            "Excerpt": highlight_keywords(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content, prompt)
                        }
                        for doc in sources
                    ]
                    st.table(source_data)

                # Dynamic follow-up questions with clickable buttons
                st.markdown("**Suggested Follow-Up Questions:**")
                for question in generate_follow_up_questions(prompt):
                    if st.button(question, key=f"followup_{question}_{len(st.session_state.messages)}"):
                        st.session_state.follow_up_question = question
                        st.experimental_rerun()

                # Handle follow-up question from button click
                if "follow_up_question" in st.session_state:
                    prompt = st.session_state.follow_up_question
                    st.chat_message('user').markdown(prompt)
                    st.session_state.messages.append({'role': 'user', 'content': prompt})
                    with st.spinner("🤖 ASKmYDocs is thinking..."):
                        time.sleep(1)
                        response = qa_chain({"query": prompt, "history": history})
                        result = response["result"].strip()
                        sources = response["source_documents"]
                        final_response = f"**Answer:** {result}" if "I don't know" not in result else f"**Answer:** {result}\n\n**Tip:** Try rephrasing your question."
                        st.chat_message('assistant').markdown(final_response)
                        st.session_state.messages.append({'role': 'assistant', 'content': final_response})
                        if sources:
                            st.markdown("**Source Documents:**")
                            source_data = [
                                {
                                    "File": os.path.basename(doc.metadata.get('source', 'Unknown')),
                                    "Page": doc.metadata.get('page', 'N/A'),
                                    "Excerpt": highlight_keywords(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content, prompt)
                                }
                                for doc in sources
                            ]
                            st.table(source_data)
                        for question in generate_follow_up_questions(prompt):
                            if st.button(question, key=f"followup_{question}_{len(st.session_state.messages)}"):
                                st.session_state.follow_up_question = question
                                st.experimental_rerun()
                        del st.session_state.follow_up_question

                # Copy and regenerate buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.button("📋 Copy Answer", on_click=lambda: st.write(f"Copied: {result}"), key=f"copy_{len(st.session_state.messages)}")
                with col2:
                    if st.button("🔄 Regenerate Answer", key=f"regen_{len(st.session_state.messages)}"):
                        with st.spinner("🤖 Regenerating answer..."):
                            time.sleep(1)
                            response = qa_chain({"query": prompt, "history": history})
                            result = response["result"].strip()
                            sources = response["source_documents"]
                            final_response = f"**Answer (Regenerated):** {result}"
                            st.chat_message('assistant').markdown(final_response)
                            st.session_state.messages.append({'role': 'assistant', 'content': final_response})
                            if sources:
                                st.markdown("**Source Documents (Regenerated):**")
                                source_data = [
                                    {
                                        "File": os.path.basename(doc.metadata.get('source', 'Unknown')),
                                        "Page": doc.metadata.get('page', 'N/A'),
                                        "Excerpt": highlight_keywords(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content, prompt)
                                    }
                                    for doc in sources
                                ]
                                st.table(source_data)

        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            st.button("🔄 Retry", on_click=lambda: None, key=f"retry_{len(st.session_state.messages)}")

    # Download chat history
    if st.session_state.messages:
        chat_history = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
        st.download_button("📥 Download Chat History", chat_history, file_name="chat_history.txt", help="Download the conversation as a text file.")
else:
    st.info("📥 Please upload PDF file(s) to begin.")