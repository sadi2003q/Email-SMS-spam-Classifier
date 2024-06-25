import streamlit as st
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import re

count_vectorisation = pickle.load(open('count_vectorisation.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)


def stem(text):
    y = []
    for i in text:
        if i not in stop_words and i not in punctuations:
            y.append(ps.stem(i.lower()))
    return y


# def Transformation(text):
#     lower = text.lower()
#     new = re.sub(r'[^a-zA-Z0-9\s]', '', lower)
#     tokenize = nltk.word_tokenize(new)
#     stemming = stem(tokenize)
#
#     return " ".join(stemming)


def Transformation(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Email/SMS spam Classifier")
st.subheader("Catch the Spam Email Based on Message of the Email/SMS")


text = st.text_area('Enter your EMAIL/SMS')


if st.button('Determine'):
    # Transformation
    transformed_Text = Transformation(text)
    # Vectorisation
    vector = count_vectorisation.transform([transformed_Text])
    # Prediction
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.subheader("Spam")
    else:
        st.subheader("Not Spam")

