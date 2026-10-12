import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
#Loading the tensorflow model for prediction
model=load_model("model.h5")

with open("token.pkl","rb")as file:
    tokenizer=pickle.load(file)
st.title("Twitter Tweets Sentiment Analysis")
tweet=st.text_area("Enter the tweet")

if st.button("Predict Sentiment") and tweet.strip():
    sequnces=tokenizer.texts_to_sequences([tweet])
    sequnces=pad_sequences(sequnces,padding="post",maxlen=99)\

    prediction=model.predict(sequnces)
    predicted_class=np.argmax(prediction,axis=1)[0]

    sentiment_map={0:"Negative",1:"Neutral",2:"Positive"}

    st.write("Sentiment",sentiment_map[predicted_class])