import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import pptx
import pandas as pd
import os
import time
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer, AutoModel
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

st.set_page_config(
    page_title="SDLC Automate APP",
    page_icon="images/favicon.png"
)

# Hide Streamlit branding, menu, and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;} /* Hides the "Manage app" menu */
    footer {visibility: hidden;} /* Hides the Streamlit footer */
    header {visibility: hidden;} /* Hides the header */
    ._link_gzau3_10 {display: none;} /* Hides "Hosted with Streamlit" */
    .stDeployButton {display: none !important;} /* Hides the deploy button */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Get the Hugging Face API key from Streamlit secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# Streamlit sidebar setup
with st.sidebar:
    st.title("Your BRD Documents")
    uploaded_file = st.file_uploader("Upload a file to generate user stories", type=["pdf", "docx", "txt", "xlsx", "pptx"])

# Function to extract text from various file types
@st.cache_resource
def extract_text_from_file(file):
    """Extracts text based on file type, with caching for faster retrieval."""
    text = ""
    file_ext = os.path.splitext(file.name)[1].lower()

    # Handle PDF files
    if file_ext == ".pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Handle Word (.docx) files
    elif file_ext == ".docx":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    # Handle text (.txt) files
    elif file_ext == ".txt":
        text = file.read().decode("utf-8")

    # Handle Excel files (.xlsx, .xls)
    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file)
        text = df.to_string()

    # Handle PowerPoint files (.pptx, .ppt)
    elif file_ext in [".pptx", ".ppt"]:
        ppt = pptx.Presentation(file)
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

    return text

# Process the uploaded file and extract text for the vector store
@st.cache_resource
def process_uploaded_file(uploaded_file):
    return extract_text_from_file(uploaded_file) if uploaded_file else ""

# Function to create embeddings using Hugging Face model
@st.cache_resource
def create_embeddings(text):
    """Creates embeddings using a Sentence Transformer model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Use a model designed for embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use `pooler_output` if available, else use mean of last_hidden_state
    if hasattr(outputs, "pooler_output"):
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.detach().numpy()

# Function for similarity search
def similarity_search(query, embeddings, top_k=3):
    query_embedding = create_embeddings(query)
    similarities = cosine_similarity(query_embedding, embeddings)
    top_k_idx = np.argsort(similarities[0])[::-1][:top_k]
    return top_k_idx

# Streamlit app setup
st.header("BRD to User Story, Test Case, Cucumber Script, and Selenium Script")

# Set up tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["User Story Generation", "User Story to Test Case", "Test Case to Cucumber Script", "Test Case to Selenium Script"])

# User Story Generation Tab
with tab1:
    start_time = time.time()
    if uploaded_file:
        text = process_uploaded_file(uploaded_file)
        if text:
            # Create embeddings for the document
            embeddings = create_embeddings(text)

            prompt_message = (
                "Think of yourself as a senior business analyst. Your responsibility is to read the Business Requirement Document "
                "and write the User Stories according to that BRD. Think step-by-step and write all possible user stories "
                "for the Business Requirement Document."
                "Make sure to give fully complete user stories."
            )

            # Perform similarity search
            start_query_time = time.time()
            top_k_idx = similarity_search(prompt_message, embeddings, top_k=3)
            st.write(f"Top 3 similar results: {top_k_idx}")

            # Use Hugging Face's pipeline for question answering (or other NLP tasks)
            qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", use_auth_token=HUGGING_FACE_API_KEY)
            response = qa_pipeline(question=prompt_message, context=text)
            st.write(response['answer'])

            # Display timing info for performance insights
            st.write(f"Document loading time: {time.time() - start_time:.2f} seconds")
            st.write(f"Query processing time: {time.time() - start_query_time:.2f} seconds")
    else:
        st.write("Please upload a BRD document in the sidebar to generate user stories.")
