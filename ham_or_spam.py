import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = [
    ("ham", "Don't forget to pick up some groceries on your way home."),
    ("spam", "Your account has been compromised. Verify your details immediately."),
    ("spam", "You have received a $500 gift card. Claim it now before it expires!"),
    ("spam", "URGENT: Your loan approval is pending. Call us now!"),
    ("ham", "Happy birthday! Hope you have a great day ahead."),
    ("ham", "Looking forward to our meeting next week."),
    ("spam", "Limited time offer! Buy now and get 50% off."),
    ("spam", "Congratulations! You've won a free vacation. Click here to claim now."),
    ("ham", "Can you send me the report by EOD?"),
    ("ham", "Hey, are we still on for lunch tomorrow?")
]

df = pd.DataFrame(data, columns=["category", "message"])
df["Spam"] = df["category"].apply(lambda x: 1 if x == "spam" else 0)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["Spam"]
model = MultinomialNB()
model.fit(X, y)

st.title("Email Spam Detector")
st.write("Enter an email message below to check if it's spam or ham.")

email_text = st.text_area("Email Content:", "")

if st.button("Check Email"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        email_vectorized = vectorizer.transform([email_text])
        
        prediction = model.predict(email_vectorized)
        
        if prediction[0] == 1:
            st.error("🚩 This email is classified as SPAM!")
        else:
            st.success("✅ This email is classified as HAM (not spam).")
