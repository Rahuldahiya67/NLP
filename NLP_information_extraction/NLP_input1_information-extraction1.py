import os
import spacy
import streamlit as st
from pdfminer.high_level import extract_text
from spacy.cli import download

# Download spaCy model if not installed
if not spacy.util.is_package("en_core_web_sm"):
    download("en_core_web_sm")

# Set spaCy model path using environment variable
os.environ["SPACY_DATA"] = "NLP_information_extraction/model.pkl" # Replace with the actual path where the models are stored on your system

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Set PDF file path
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# If PDF file is uploaded
if pdf_file is not None:
    # Extract text from PDF file
    text = extract_text(pdf_file)
    
    # Create a spaCy object from the text
    doc = nlp(text)

    # Display entities in Streamlit
    for ent in doc.ents:
        st.write(ent.text, ent.label_)
