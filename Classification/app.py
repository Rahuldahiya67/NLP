import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle

# Read the dataset
bbc_text = pd.read_csv("Classification/bbc-text.txt")
bbc_text = bbc_text.rename(columns={'text': 'News_Headline'}, inplace=False)
bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})

# Split the data
X = bbc_text.News_Headline
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
st.set_page_config(page_title="BBC News Category Classification", page_icon=":newspaper:", layout="wide")
st.title("BBC News Category Classification")

input_text = st.text_area("Enter the news headline", value="", height=100)
if st.button("Check Category"):
    input_text_transformed = vector.transform([input_text]).toarray()
    prediction = naivebayes.predict(input_text_transformed)[0]
    prediction_mapping = {
        0: 'Tech',
        1: 'Business',
        2: 'Sports',
        3: 'Entertainment',
        4: 'Politics',
        5: 'Health',
        6: 'Education',
        7: 'Environment',
        8: 'Travel',
        9: 'Food',
        10: 'Fashion'
    }
    result = prediction_mapping[prediction]
    st.success(f"Predicted category: {result}")

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
