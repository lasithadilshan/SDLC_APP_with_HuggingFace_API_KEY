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
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceLLM

st.set_page_config(
    page_title="SDLC Automate APP",
    page_icon="images/favicon.png"
)

# Get the API key from Streamlit secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# Streamlit sidebar setup
with st.sidebar:
    st.title("Your BRD Documents")
    uploaded_file = st.file_uploader("Upload a file to generate user stories", 
                                     type=["pdf", "docx", "txt", "xlsx", "pptx"])

# Function to extract text from various file types
@st.cache_resource
def extract_text_from_file(file):
    """Extracts text based on file type."""
    text = ""
    file_ext = os.path.splitext(file.name)[1].lower()

    if file_ext == ".pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

    elif file_ext == ".docx":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif file_ext == ".txt":
        text = file.read().decode("utf-8")

    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file)
        text = df.to_string()

    elif file_ext == ".pptx":
        ppt = pptx.Presentation(file)
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

    return text.strip()

# Function to process uploaded file
@st.cache_resource
def process_uploaded_file(uploaded_file):
    return extract_text_from_file(uploaded_file) if uploaded_file else ""

# Function to create vector store
@st.cache_resource
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(api_key=HUGGINGFACE_API_KEY)
    return FAISS.from_texts(chunks, embeddings)

# Initialize LLM
llm = HuggingFaceLLM(
    api_key=HUGGINGFACE_API_KEY,
    model_name="HuggingFace/transformers-gpt",
    temperature=0.7,
    max_tokens=500
)

# Streamlit app setup
st.header("BRD to User Story, Test Case, Cucumber Script, and Selenium Script")

# Set up tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "User Story Generation", "User Story to Test Case", 
    "Test Case to Cucumber Script", "Test Case to Selenium Script"
])

# User Story Generation Tab
with tab1:
    if uploaded_file:
        start_time = time.time()
        text = process_uploaded_file(uploaded_file)
        
        if text:
            vector_store = create_vector_store(text)
            retriever = vector_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            prompt_message = (
                "You are a senior business analyst. Read the Business Requirement Document "
                "and generate user stories based on its content. Think step-by-step."
            )

            start_query_time = time.time()
            response = qa_chain.invoke({"query": prompt_message})
            result = response.get("result", "No response generated.")

            st.write(result)
            st.write(f"Processing Time: {time.time() - start_query_time:.2f} sec")
    else:
        st.write("Upload a BRD document to generate user stories.")

# User Story to Test Case Tab
with tab2:
    st.subheader("Convert User Story to Test Case")
    user_story_text = st.text_area("Enter the user story text:")

    if st.button("Generate Test Cases"):
        if user_story_text:
            test_case_prompt = (
                "You are a senior QA engineer. Read the user story below and generate structured test cases, "
                "covering functional and edge cases.\n\n" + user_story_text
            )
            response = qa_chain.invoke({"query": test_case_prompt})
            result = response.get("result", "No test cases generated.")
            st.write(result)
        else:
            st.write("Enter a user story to generate test cases.")

# Test Case to Cucumber Script Tab
with tab3:
    st.subheader("Convert Test Case to Cucumber Script")
    test_case_text = st.text_area("Enter the test case text:")

    if st.button("Generate Cucumber Script"):
        if test_case_text:
            cucumber_prompt = (
                "Convert the following test case into a Cucumber script using Gherkin syntax (Given-When-Then format).\n\n"
                + test_case_text
            )
            response = qa_chain.invoke({"query": cucumber_prompt})
            result = response.get("result", "No Cucumber script generated.")
            st.write(result)
        else:
            st.write("Enter a test case to generate a Cucumber script.")

# Test Case to Selenium Script Tab
with tab4:
    st.subheader("Convert Test Case to Selenium Script")
    selenium_test_case = st.text_area("Enter the test case text:")

    if st.button("Generate Selenium Script"):
        if selenium_test_case:
            selenium_prompt = (
                "Convert the following test case into a Selenium automation script using Python Selenium WebDriver.\n\n"
                + selenium_test_case
            )
            response = qa_chain.invoke({"query": selenium_prompt})
            result = response.get("result", "No Selenium script generated.")
            st.write(result)
        else:
            st.write("Enter a test case to generate a Selenium script.")
