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
with open("model1.pkl", "rb") as f:
    ml_model = pickle.load(f)

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
