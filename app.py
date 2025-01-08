import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pptx
import pandas as pd
import os
import time
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceLLM

# Set up Streamlit page
st.set_page_config(
    page_title="SDLC Automate APP",
    page_icon="📄"
)

# Check for API Key
if "HUGGINGFACE_API_KEY" not in st.secrets:
    st.error("API key not found! Please add 'HUGGINGFACE_API_KEY' in Streamlit Secrets.")
    st.stop()

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# Sidebar for file upload
with st.sidebar:
    st.title("Upload BRD Document")
    uploaded_file = st.file_uploader(
        "Upload a file to generate user stories",
        type=["pdf", "docx", "txt", "xlsx", "pptx"]
    )

# Function to extract text from various file types
@st.cache_resource
def extract_text_from_file(file):
    """Extracts text based on file type."""
    text = ""
    file_ext = os.path.splitext(file.name)[1].lower()

    try:
        if file_ext == ".pdf":
            pdf_reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])

        elif file_ext == ".docx":
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif file_ext == ".txt":
            text = file.read().decode("utf-8")

        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
            text = df.to_string()

        elif file_ext in [".pptx", ".ppt"]:
            ppt = pptx.Presentation(file)
            text = "\n".join([shape.text for slide in ppt.slides for shape in slide.shapes if hasattr(shape, "text")])
    
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    
    return text.strip()

# Process the uploaded file and extract text for the vector store
@st.cache_resource
def process_uploaded_file(uploaded_file):
    return extract_text_from_file(uploaded_file) if uploaded_file else ""

# Function to create vector store from extracted text
@st.cache_resource
def create_vector_store(text):
    if not text:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=800,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(huggingface_api_key=HUGGINGFACE_API_KEY)
    return FAISS.from_texts(chunks, embeddings)

# Streamlit app setup
st.header("BRD Automation: User Stories, Test Cases, & Scripts")

# Set up tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "User Story Generation",
    "User Story to Test Case",
    "Test Case to Cucumber Script",
    "Test Case to Selenium Script"
])

# Load vector store if file is uploaded
if uploaded_file:
    text = process_uploaded_file(uploaded_file)
    
    if text:
        vector_store = create_vector_store(text)

        # Initialize LLM
        llm = HuggingFaceLLM(
            huggingface_api_key=HUGGINGFACE_API_KEY,
            temperature=0.7,
            max_tokens=500,
            model_name="HuggingFace/transformers-gpt"
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
    else:
        vector_store = None

# User Story Generation Tab
with tab1:
    if uploaded_file and vector_store:
        st.subheader("Generated User Stories")
        
        start_time = time.time()
        
        prompt_message = (
            "You are a senior business analyst. Read the Business Requirement Document "
            "and generate user stories based on it. Provide structured and detailed user stories."
        )

        start_query_time = time.time()
        response = qa_chain.invoke({"query": prompt_message})
        
        st.write(response["result"])
        st.write(f"Document loading time: {time.time() - start_time:.2f} seconds")
        st.write(f"Query processing time: {time.time() - start_query_time:.2f} seconds")
    else:
        st.warning("Please upload a BRD document in the sidebar to generate user stories.")

# User Story to Test Case Tab
with tab2:
    st.subheader("Convert User Story to Test Case")
    user_story_text = st.text_area("Enter a user story:")

    if st.button("Generate Test Cases"):
        if user_story_text and vector_store:
            response = qa_chain.invoke({"query": f"Convert this user story into test cases: {user_story_text}"})
            st.write(response["result"])
        else:
            st.warning("Please enter a user story.")

# Test Case to Cucumber Script Tab
with tab3:
    st.subheader("Convert Test Case to Cucumber Script")
    test_case_text = st.text_area("Enter a test case:")

    if st.button("Generate Cucumber Script"):
        if test_case_text and vector_store:
            response = qa_chain.invoke({"query": f"Convert this test case into a Cucumber script: {test_case_text}"})
            st.write(response["result"])
        else:
            st.warning("Please enter a test case.")

# Test Case to Selenium Script Tab
with tab4:
    st.subheader("Convert Test Case to Selenium Script")
    selenium_test_case = st.text_area("Enter a test case:")

    if st.button("Generate Selenium Script"):
        if selenium_test_case and vector_store:
            response = qa_chain.invoke({"query": f"Convert this test case into a Python Selenium script: {selenium_test_case}"})
            st.write(response["result"])
        else:
            st.warning("Please enter a test case.")
