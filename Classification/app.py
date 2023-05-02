import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle

# Read the dataset
bbc_text = pd.read_csv("Classification/bbc-text.txt")
bbc_text = bbc_text.rename(columns={'text': 'News Headline'}, inplace=False)
bbc_text.category = bbc_text.category.map({'tech': 0, 'business': 1, 'sport': 2, 'entertainment': 3, 'politics': 4})

# Split the data
X = bbc_text['News Headline']
y = bbc_text.category
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)

# Vectorize the text
vector = CountVectorizer(stop_words='english', lowercase=False)
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_test_transformed = vector.transform(X_test)

# Train the model
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(naivebayes, f)

# Define the Streamlit app
st.set_page_config(page_title="BBC Text Classification", page_icon=":books:")
st.title("BBC Text Classification")
st.write("Enter a news headline below to classify its category")

input_text = st.text_area("News Headline", value="", height=100)
if st.button("Classify", key='classify'):
    if input_text.strip() == "":
        st.error("Please enter a news headline.")
    else:
        input_text_transformed = vector.transform([input_text]).toarray()
        prediction = naivebayes.predict(input_text_transformed)[0]
        prediction_mapping = {0: 'TECH', 1: 'BUSINESS', 2: 'SPORTS', 3: 'ENTERTAINMENT', 4: 'POLITICS'}
        result = prediction_mapping[prediction]
        st.success(f"Predicted category: {result}")
