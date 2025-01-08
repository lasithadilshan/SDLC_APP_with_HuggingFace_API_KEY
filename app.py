import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pptx
import pandas as pd
import os
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(
    page_title="SDLC Automate APP",
    page_icon="images/favicon.png"
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    ._link_gzau3_10 {display: none;}
    .stDeployButton {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

with st.sidebar:
    st.title("Your BRD Documents")
    uploaded_file = st.file_uploader("Upload a file to generate user stories", type=["pdf", "docx", "txt", "xlsx", "pptx"])

def extract_text_from_file(file):
    text = ""
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext == ".pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    elif file_ext == ".docx":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_ext == ".txt":
        text = file.read().decode("utf-8")
    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file)
        text = df.to_string()
    elif file_ext in [".pptx", ".ppt"]:
        ppt = pptx.Presentation(file)
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    return text

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=800,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]
    return FAISS.from_texts(chunks, embedding_model)

st.header("BRD to User Story, Test Case, Cucumber Script, and Selenium Script")

tab1, tab2, tab3, tab4 = st.tabs(["User Story Generation", "User Story to Test Case", "Test Case to Cucumber Script", "Test Case to Selenium Script"])

with tab1:
    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
        if text:
            vector_store = create_vector_store(text)
            prompt_message = "Think of yourself as a senior business analyst. Read the BRD and write all possible user stories."
            matches = vector_store.similarity_search(prompt_message, k=3)
            response = llm(prompt_message, max_length=500, do_sample=True)[0]['generated_text']
            st.write(response)
    else:
        st.write("Please upload a BRD document in the sidebar to generate user stories.")

with tab2:
    user_story_text = st.text_area("Enter the user story text here to generate test cases:")
    if st.button("Generate Test Cases"):
        if user_story_text:
            test_case_prompt = "Generate test cases for: " + user_story_text
            response = llm(test_case_prompt, max_length=500, do_sample=True)[0]['generated_text']
            st.write(response)

with tab3:
    test_case_text = st.text_area("Enter the test case text here to generate Cucumber script:")
    if st.button("Generate Cucumber Script"):
        if test_case_text:
            cucumber_prompt = "Convert the following test case into a Cucumber script: " + test_case_text
            response = llm(cucumber_prompt, max_length=500, do_sample=True)[0]['generated_text']
            st.write(response)

with tab4:
    selenium_test_case_text = st.text_area("Enter the test case text here to generate Selenium script:")
    if st.button("Generate Selenium Script"):
        if selenium_test_case_text:
            selenium_prompt = "Convert the following test case into a Selenium WebDriver script: " + selenium_test_case_text
            response = llm(selenium_prompt, max_length=500, do_sample=True)[0]['generated_text']
            st.write(response)
