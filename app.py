import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.set_page_config(page_title="Stress Detector", page_icon="😰", layout="centered")

st.title("😰 Stress Detection from Social Media Text")
st.write("Enter a Reddit or Twitter post below to check if it shows signs of stress.")

user_input = st.text_area("🗒️ Enter text here:", height=150)

if st.button("Analyze"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == "stress":
            st.error("🚨 This post indicates **Stress**.")
        else:
            st.success("😊 This post seems **Non-Stressful**.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
