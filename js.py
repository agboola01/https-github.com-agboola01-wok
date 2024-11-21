import os
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from fpdf import FPDF
import pdfplumber
from docx import Document

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Ensure the "combine" folder exists for storing uploaded CVs
COMBINE_FOLDER = "combine"
os.makedirs(COMBINE_FOLDER, exist_ok=True)

# Function to extract text from a PDF file
def extract_pdf_text(file_path):
    content = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            content += page_text
    return content

# Function to extract text from a Word file
def extract_word_text(file_path):
    doc = Document(file_path)
    content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return content

# Function to save and process uploaded CVs
def save_and_extract_cv(uploaded_file):
    # Save the file to the combine folder
    file_path = os.path.join(COMBINE_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text based on file type
    if file_path.endswith(".pdf"):
        content = extract_pdf_text(file_path)
    elif file_path.endswith(".docx"):
        content = extract_word_text(file_path)
    else:
        st.error("Unsupported file format! Please upload a PDF or DOCX file.")
        return None, file_path
    
    return content, file_path

# Function to process job descriptions from the jd folder
def process_job_descriptions(folder="jd"):
    jd_texts = []
    filenames = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if file_path.endswith(".pdf"):
            content = extract_pdf_text(file_path)
            jd_texts.append(content)
            filenames.append(filename)
    return jd_texts, filenames

# Function to embed and compute cosine similarities between a CV and job descriptions
def compute_similarities(cv_text, jd_texts):
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_texts, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(cv_embedding, jd_embeddings)[0].cpu().numpy()
    return similarities

# Streamlit UI
st.title("Upload Your CV to Find Best Job Matches")
st.header("Find the Best Job Descriptions for Your CV")

# File uploader for CV
uploaded_file = st.file_uploader("Upload Your CV (PDF or DOCX)", type=["pdf", "docx"])

# Initialize the number of matches to display
top_n = st.number_input("Number of Matches to Display", min_value=1, max_value=10, value=5, step=1)

if uploaded_file and st.button("Find Best Job Matches"):
    # Save and extract text from the uploaded CV
    cv_text, saved_file_path = save_and_extract_cv(uploaded_file)

    if cv_text:
        st.success(f"Your CV has been saved in the '{COMBINE_FOLDER}' folder as '{uploaded_file.name}'")
        
        # Process job descriptions from the folder
        jd_folder = "jd"
        jd_texts, jd_filenames = process_job_descriptions(jd_folder)

        if not jd_texts:
            st.warning("No job descriptions found in the 'jd' folder.")
        else:
            # Compute similarities between the CV and job descriptions
            similarities = compute_similarities(cv_text, jd_texts)

            # Display results
            top_indices = np.argsort(-similarities)[:top_n]  # Get top N indices
            st.subheader(f"Top {top_n} Matching Job Descriptions")
            for idx in top_indices:
                match_percentage = similarities[idx] * 100  # Convert similarity to percentage
                st.write(f"Match Score: {match_percentage:.2f}%")
                
                # Display the job description content
                job_description_content = jd_texts[idx]
                st.text_area(f"Job Description Content ({jd_filenames[idx]})", job_description_content, height=200)
                
                # Provide download option for the matched job description
                jd_file_path = os.path.join(jd_folder, jd_filenames[idx])
                st.download_button(
                    label=f"Download {jd_filenames[idx]}",
                    data=open(jd_file_path, "rb").read(),
                    file_name=jd_filenames[idx],
                    mime="application/pdf"
                )
