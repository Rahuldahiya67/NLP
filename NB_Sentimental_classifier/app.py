import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle

# Define the Streamlit app
st.set_page_config(page_title="classifier", page_icon=":newspaper:", layout="wide")
st.title("classifier")

# Define the trainmodel function
def trainmodel():
    #load data
    review = pd.read_csv('reviews.csv')
    review = bbc_text.rename(columns = {'text': 'review'}, inplace = False)
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(review.review, review.polarity, train_size = 0.6, random_state = 1)
    
    #fit vectorizer on training data
    vector = CountVectorizer(stop_words='english', lowercase=False)
    vector.fit(X_train)
    
    #transform data for training and testing
    X_train_transformed = vector.transform(X_train)
    X_test_transformed = vector.transform(X_test)
    
    #train naive bayes classifier
    naivebayes = MultinomialNB()
    naivebayes.fit(X_train_transformed, y_train)
    
    #evaluate classifier
    accuracy = naivebayes.score(X_test_transformed, y_test)
    st.write('Accuracy:', accuracy)
    
    #save model
    with open('saved_model.pkl', 'wb') as file:
        pickle.dump(naivebayes, file)

# Define the predict function
def predict():
    #load saved model
    with open('saved_model.pkl', 'rb') as file:
        saved_model = pickle.load(file)
    
    #get user input
    user_input = st.text_input('Enter a review:')
    
    #make prediction
    if user_input:
        review = [user_input]
        vector = CountVectorizer(stop_words='english', lowercase=False)
        with open('vector.pkl', 'wb') as file:
            pickle.dump(vector, file)
        vector.fit(review)
        vec = vector.transform(review).toarray()
        prediction = saved_model.predict(vec)[0]
        if prediction == 1:
            st.write('The review is positive.')
        else:
            st.write('The review is negative.')

# Define the Streamlit app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Homepage", "Train Model", "Make Prediction"])
    
    if app_mode == "Homepage":
        st.write("Welcome to the classifier app!")
    
    elif app_mode == "Train Model":
        trainmodel()
    
    elif app_mode == "Make Prediction":
        predict()

if __name__ == "__main__":
    main()
