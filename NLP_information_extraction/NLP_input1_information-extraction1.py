#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import spacy
import streamlit as st
from pdfminer.high_level import extract_text

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define function for extracting entities
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

# Define Streamlit app
def main():
    # Set app title
    st.title("PDF Entity Recognition")

    # Add file uploader
    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("", type="pdf")

    # Process uploaded file
    if uploaded_file is not None:
        # Extract text from PDF
        text = extract_text(uploaded_file)
        # Display text
        st.subheader("PDF Text:")
        st.write(text)
        # Extract entities from text
        entities = extract_entities(text)
        # Display entities
        st.subheader("Entities:")
        for entity in entities:
            st.write(entity)

# Run app
if __name__ == "__main__":
    main()


# In[ ]:




