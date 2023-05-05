import pandas as pd
import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
review = pd.read_csv('reviews.csv')
review = review.rename(columns={'text': 'review'})

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X = review.review
y = review.polarity
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)

# Define the trainmodel function to train the model and save it to a file
def trainmodel():
    # Train the vectorizer and classifier
    vector = CountVectorizer(stop_words='english', lowercase=False)
    vector.fit(X_train)
    X_train_transformed = vector.transform(X_train)
    naivebayes = MultinomialNB()
    naivebayes.fit(X_train_transformed, y_train)

    # Save the model to a file
    with open('saved_model.pkl', 'wb') as f:
        pickle.dump((vector, naivebayes), f)

# Define the predict function to take user input and predict the sentiment
def predict():
    # Load the saved model from the file
    with open('saved_model.pkl', 'rb') as f:
        vector, naivebayes = pickle.load(f)

    # Get the user input
    input_text = st.text_input('Enter your review')

    # Make a prediction
    if st.button('Predict'):
        vec = vector.transform([input_text]).toarray()
        prediction = naivebayes.predict(vec)[0]
        if prediction == 0:
            st.write('Negative')
        else:
            st.write('Positive')

# Call the trainmodel function to train and save the model
trainmodel()

# Create the Streamlit app
st.set_page_config(page_title='Sentiment Analysis', page_icon=':smiley:')
st.title('Sentiment Analysis')
predict()
