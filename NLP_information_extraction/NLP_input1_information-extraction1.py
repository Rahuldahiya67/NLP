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

# Set up some CSS styles
main_css = """
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem;
"""

title_css = """
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 2rem;
"""

upload_css = """
    padding: 1rem;
    border: 2px dashed #ccc;
    text-align: center;
"""

# Define the Streamlit app
def app():
    # Set the page title and add some CSS styles
    st.set_page_config(page_title="PDF Entity Extraction", page_icon="📚")
    st.markdown(f'<style>{main_css}</style>', unsafe_allow_html=True)
    
    # Add a title
    st.markdown('<h1 class="title" style="{}">PDF Entity Extraction</h1>'.format(title_css), unsafe_allow_html=True)
    
    # Add an upload button
    st.markdown('<div class="upload" style="{}"><p>Upload a PDF file to extract entities:</p><br><br><br><br></div>'.format(upload_css), unsafe_allow_html=True)
    
    # Set PDF file path
    pdf_file = st.file_uploader(" ", type="pdf")

    # If PDF file is uploaded
    if pdf_file is not None:
        # Extract text from PDF file
        text = extract_text(pdf_file)
        
        # Create a spaCy object from the text
        doc = nlp(text)

        # Display entities in Streamlit
        for ent in doc.ents:
            st.write(ent.text, ent.label_)
