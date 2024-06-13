import os
import shutil
import streamlit as st
from ingest import load_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from tempfile import NamedTemporaryFile

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<YOUR_API_KEY>"
os.environ["LANGCHAIN_PROJECT"] = "<YOUR_PROJECT_NAME>"
os.environ["EMBEDDINGS_MODEL_NAME"] = "all-MiniLM-L6-v2"
os.environ["PERSIST_DIRECTORY"] = "db"
os.environ["TARGET_SOURCE_CHUNKS"] = "4"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '<YOUR_API_TOKEN>'  # Add your token here

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=os.environ.get("EMBEDDINGS_MODEL_NAME"))
    vectorstore = Chroma.from_documents(text_chunks, embeddings, persist_directory=os.environ.get("PERSIST_DIRECTORY"))
    return vectorstore

def handle_userinput(user_question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents(user_question)
    return docs

def main():
    st.set_page_config(page_title="Chat with your documents", page_icon=":books:")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with your documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.vectorstore is not None:
        docs = handle_userinput(user_question, st.session_state.vectorstore)
        st.write(f"**Question:** {user_question}")
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            st.write(f"> **{source}:** {doc.page_content}")

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if uploaded_files:
                source_directory = "source_documents"
                if not os.path.exists(source_directory):
                    os.makedirs(source_directory)
                for uploaded_file in uploaded_files:
                    temp_file = NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    shutil.move(temp_file.name, os.path.join(source_directory, uploaded_file.name))

                st.write("Processing documents...")
                documents = load_documents(source_directory)
                if not documents:
                    st.error("No documents were loaded.")
                    return

                text_chunks = get_text_chunks(documents)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.success("Documents processed successfully!")
            else:
                st.error("Please upload at least one document.")

if __name__ == '__main__':
    main()
