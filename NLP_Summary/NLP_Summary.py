# Import libraries
import streamlit as st
import pdfminer.high_level
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import pickle

# Load pre-trained model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load the saved machine learning model
with open("model.pkl", "rb") as f:
    ml_model = pickle.load(f)

# Define a function to extract summary from a PDF file
def extract_summary(pdf_file):
    # Extract text from pdf file
    text = pdfminer.high_level.extract_text(pdf_file)
    # Tokenize text and generate summary
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"])
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Return the summary
    return summary

# Define the title of the app
st.title("PDF Summary Extractor")

# Define a file uploader for the user to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# If the user uploads a file, run the machine learning model and show the result
if uploaded_file is not None:
    # Extract summary from the PDF file
    summary = extract_summary(uploaded_file)
    # Display the summary
    st.write(summary)
