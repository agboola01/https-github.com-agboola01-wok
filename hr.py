import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from fpdf import FPDF
from io import BytesIO
from unidecode import unidecode
import pdfplumber
from docx import Document
# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Define heading keywords for extracting sections from CVs
heading_keywords = { 
    "Education": ["Education", "Degree", "University", "School"],
    "Skills": ["Skills", "Skillset", "Technical Skills", "Core Competencies"],
    "Experience/project": ["Experience", "Work Experience", "Professional Experience", "Projects", "Project", "Employment"],
    "Certifications": ["Certifications", "Certification", "License"]
}
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
# Function to extract specific sections from CV content
def extract_sections(content):
    sections = {key: "" for key in heading_keywords.keys()}
    current_section = None
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if any(keyword.lower() in line.lower() for keyword in heading_keywords["Education"]):
            current_section = "Education"
        elif any(keyword.lower() in line.lower() for keyword in heading_keywords["Skills"]):
            current_section = "Skills"
        elif any(keyword.lower() in line.lower() for keyword in heading_keywords["Experience/project"]):
            current_section = "Experience/project"
        elif any(keyword.lower() in line.lower() for keyword in heading_keywords["Certifications"]):
            current_section = "Certifications"
        if current_section:
            sections[current_section] += line + "\n"
    return sections
# Function to process CVs from the directory
def process_cvs(directory):
    cv_data = []
    filenames = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if file_path.endswith(".pdf"):
            content = extract_pdf_text(file_path)
        elif file_path.endswith(".docx"):
            content = extract_word_text(file_path)
        else:
            continue
        sections = extract_sections(content)
        filenames.append(filename)
        cv_data.append(sections)
    return cv_data, filenames
# Function to embed and compute cosine similarities between job description and CVs
def compute_similarities(job_text, experience_texts):
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    experience_embeddings = model.encode(experience_texts, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(job_embedding, experience_embeddings)[0].cpu().numpy()
    return similarities
# Streamlit UI
st.title("Job Description - CV Matching System")
st.header("Match Job Descriptions to CVs")
# Input for job description
job_title = st.text_input("Position Title", placeholder="e.g., Data Scientist")
company_name = st.text_input("Company Name", placeholder="e.g., TechCorp")
job_description = st.text_area("Job Description", placeholder="Enter the job description here...")
# Initialize the number of matches to display
top_n = st.number_input("Number of Matches to Display", min_value=1, max_value=10, value=5, step=1)
if job_description and st.button("Find Best Matches"):
    # Combine job details into a single text
    job_text = f"{job_title} at {company_name}: {job_description}"
    # Process CVs from the directory
    cv_directory = "combined"  # Path where your CVs are stored
    cv_data, filenames = process_cvs(cv_directory)
    # Extract the "Experience/Project" text from each CV
    experience_texts = [cv['Experience/project'] for cv in cv_data]
    # Compute similarities between the job description and the "Experience/project" sections
    similarities = compute_similarities(job_text, experience_texts)
    # Display results
    top_indices = np.argsort(-similarities)[:top_n]  # Get top N indices
    st.subheader(f"Top {top_n} Matching CVs")
    for idx in top_indices:
        match_percentage = similarities[idx] * 100  # Convert similarity to percentage
        st.write(f"Match Score: {match_percentage:.2f}%")
        # Display the "Experience/project" section from the CV
        experience_content = cv_data[idx]["Experience/project"]
        st.text_area(f"CV Content ({filenames[idx]})", experience_content, height=200)
        # Provide download option for the original CV
        cv_file_path = os.path.join(cv_directory, filenames[idx])
        st.download_button(
            label=f"Download Original {filenames[idx]}",
            data=open(cv_file_path, "rb").read(),
            file_name=filenames[idx],
            mime="application/octet-stream"  # Generic file mime type
        )
