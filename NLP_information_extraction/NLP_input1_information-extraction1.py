#!/usr/bin/env python
# coding: utf-8

import spacy
from spacy.cli import download
import streamlit as st
from pdfminer.high_level import extract_text
import os

# Download spaCy model if not installed
if not spacy.util.is_package("en_core_web_sm"):
    download("en_core_web_sm")


# Define the Streamlit app
st.set_page_config(page_title="Information Extraction", page_icon=":extraction:", layout="wide")
st.title("Information Extraction")

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
        if ent.label_ != "CARDINAL":
            st.write(ent.text, ent.label_))
        
st.sidebar.text("Developed by Rahul")

# Add some CSS styles to make the app look more attractive
st.markdown(
"""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f8f8;
}
h1, h2, h3 {
    font-weight: bold;
    color: #333;
}
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
textarea {
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 10px;
    font-size: 16px;
    margin-bottom: 10px;
}
button {
    background-color: #333;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}
button:hover {
    background-color: #555;
}
.success {
    background-color: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
</style>
""",
unsafe_allow_html=True)
