#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import libraries
import spacy

get_ipython().system('pip install pdfminer.six  #pdfminer.six is a tool used for extracting text from PDF files.')

from pdfminer.high_level import extract_text #the "extract_text" function is used to extract the text from a PDF file and then pass it to a spaCy object for further processing.
'''
"pdfminer.high_level" is a module in the pdfminer library that provides 
a high-level interface for extracting text and other data from PDF files. 
This module contains functions that simplify the process of extracting 
text from PDF files, compared to the lower-level modules in the pdfminer 
library.
'''

# Load spaCy model
'''
This line of code loads the spaCy model "en_core_web_sm", 
which is a small English language model. This model has 
already been trained on a large corpus of English text 
and can be used to analyze the entities and relationships 
in natural language text.
'''
nlp = spacy.load("en_core_web_sm")


# Define pdf file path
pdf_file = "C:/Users/hp/Downloads/climate_change.pdf"


# Extract text from pdf file
text = extract_text(pdf_file)

# Create a spaCy object from the text
doc = nlp(text)

# Loop through the entities in the doc object
for ent in doc.ents:
  # Print the text and label of each entity
  print(ent.text, ent.label_)
'''
When we loop through the entities in the doc object using a for loop, 
we are essentially iterating over a collection of named entities that 
are present in the document. For each iteration of the loop, the ent 
variable represents a single named entity, which has two properties 
that we can access: ent.text and ent.label_.
'''


# In[ ]:




