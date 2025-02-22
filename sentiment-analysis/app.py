import pickle
import dill
import re
import contractions
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st

st.title('Sentiment Analysis')
lemma = WordNetLemmatizer()
stop_words = stopwords.words('English')
classifier = pickle.load(open('classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
slang_dict = dill.load(open('slangs.pkl', 'rb'))
preprocessing = dill.load(open('preprocessing.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

new_text = st.text_area('Enter Your Text')


def predict_new(text):
    processed_text = preprocessing(text)
    processed_text = vectorizer.transform([processed_text]).toarray()
    prediction = classifier.predict(processed_text)
    return encoder.classes_[prediction]


if st.button('Predict'):
    new_prediction = predict_new(new_text)[0]
    st.write(new_prediction.upper())
