
import nltk # Text libarary
import pandas as pd
import numpy as np
import string # Removing special characters {#, @, ...}
import re # Regex Package
from nltk.corpus import stopwords # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer # Stemmer & Lemmatizer
import pickle
import streamlit as st
from PIL import Image
import os, sys

model_path = 'C:/rf_model.pk'
vect_path = 'C:/tfidf_vectorizer.pk'
loaded_model = pickle.load(open(model_path, 'rb'))
loaded_vect =  pickle.load(open(vect_path, 'rb'))

def clean_review(text):
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    text = text.lower()
    text = text.replace('[^\w\s]', '')
    text = ' '.join([word for word in text.split() if word not in (stop_words)])
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = ' '.join([lemmatizer.lemmatize(w) for w in text])
    stemmer = SnowballStemmer("english")
    text = text.split()
    text = ' '.join([stemmer.stem(w) for w in text])
    return text   


def raw_test(review, model, vectorizer):
    # Clean Review
    review = clean_review(review)
    review_c = review
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"

def main():
      # web page title
    st.title("WElcome in Food review Prediction")
      
    html_temp = """
    <div style ="background-color:gray;padding:15px">
    <h1 style ="color:white;text-align:center;">Review prediction</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # Take the text data 
    
    review = st.text_input("Enter your review to predict : ")
    x =""
    
    if st.button("Predict"):
        x = raw_test(review, loaded_model, loaded_vect)
    st.success('The prediction is {}'.format(x))
     
if __name__=='__main__':
    main()
