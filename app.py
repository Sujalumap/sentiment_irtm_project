import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# Step 1: Sample Dataset (inbuilt)
# ----------------------------
data = {
    "review": [
        "This product is amazing",
        "Very bad quality",
        "I love this phone",
        "Worst purchase ever",
        "Not bad, okay product",
        "Excellent performance",
        "Terrible experience",
        "Good value for money",
        "Waste of money",
        "Superb and fantastic",
        "Battery is very good",
        "Very slow performance",
        "Design is okay",
        "Camera quality is awesome",
        "Not worth the price"
    ],
    "sentiment": [
        "Positive","Negative","Positive","Negative","Neutral",
        "Positive","Negative","Positive","Negative","Positive",
        "Positive","Negative","Neutral","Positive","Negative"
    ]
}

df = pd.DataFrame(data)

# ----------------------------
# Step 2: Text Cleaning
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['review'] = df['review'].apply(clean_text)

# ----------------------------
# Step 3: Model Training
# ----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review'])

model = MultinomialNB()
model.fit(X, df['sentiment'])

# ----------------------------
# Step 4: UI
# ----------------------------
st.title("📊 Sentiment Based Product Analyzer")

st.write("Type your product review below 👇")

user_input = st.text_area("Enter Review")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "Positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "Negative":
            st.error("😡 Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")
    else:
        st.warning("⚠️ Please enter a review!")

# ----------------------------
# Footer
# ----------------------------
st.write("🚀 Real-Time Sentiment Analyzer")