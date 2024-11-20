from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load job descriptions database
@st.cache
def load_job_data(file_path):
    return pd.read_csv(file_path)

# Load CV database
@st.cache
def load_cv_data(file_path):
    return pd.read_csv(file_path)

# Embed function for CVs and jobs
def embed_text(data, columns):
    texts = data[columns].apply(lambda x: ' '.join(x.dropna()), axis=1)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings, data

# Find the best matching CVs
def find_best_matches(cv_embeddings, job_embedding, top_n=5):
    similarities = util.pytorch_cos_sim(job_embedding, cv_embeddings)[0]
    top_results = np.argpartition(-similarities, range(top_n))[:top_n]
    return top_results, similarities

# Streamlit UI
st.title("Job Description - CV Matching System")
st.header("Match Job Descriptions to CVs")

# Upload job description
job_title = st.text_input("Position Title", placeholder="e.g., Data Scientist")
company_name = st.text_input("Company Name", placeholder="e.g., TechCorp")
job_description = st.text_area("Job Description", placeholder="Enter the job description here...")

if job_description and st.button("Find Best Matches"):
    # Combine the job details into a single string for embedding
    job_text = f"{job_title} at {company_name}: {job_description}"
    job_embedding = model.encode(job_text, convert_to_tensor=True)

    # Load CV database and embed CVs
    cv_data = load_cv_data('data/cv_database.csv')  # Path to your CV database
    cv_embeddings, cv_data = embed_text(cv_data, ['education', 'skills', 'projects', 'certifications', 'experience'])

    # Find top matches
    top_n = st.number_input("Number of Matches to Display", min_value=1, max_value=10, value=5, step=1)
    top_results, similarities = find_best_matches(cv_embeddings, job_embedding, top_n=top_n)

    # Display top matches
    st.subheader("Top Matching CVs")
    for idx in top_results:
        match_percentage = similarities[idx].item() * 100  # Convert similarity to percentage
        st.write(f"Match Score: {match_percentage:.2f}%")
        st.write(cv_data.iloc[idx].to_dict())

    # Allow downloading of matching CVs
    st.subheader("Download Top Matching CVs")
    for idx in top_results:
        cv_content = cv_data.iloc[idx].to_dict()
        cv_text = '\n'.join([f"{key}: {value}" for key, value in cv_content.items()])
        cv_bytes = BytesIO()
        cv_bytes.write(cv_text.encode())
        cv_bytes.seek(0)
        st.download_button(
            label=f"Download CV {idx + 1}",
            data=cv_bytes,
            file_name=f"matching_cv_{idx + 1}.txt",
            mime="text/plain"
        )
