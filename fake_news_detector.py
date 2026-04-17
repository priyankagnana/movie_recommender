# fake_news_detector.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# ==============================
# 1. Dataset
# ==============================
data = {
    "text": [
        "Government announces new policy",
        "Scientists discover cure for disease",
        "Aliens landed in New York",
        "Click here to win iPhone",
        "Breaking: major economic update",
        "Fake news spreading on social media"
    ],
    "label": [0, 0, 1, 1, 0, 1]  # 0 = real, 1 = fake
}

df = pd.DataFrame(data)

# ==============================
# 2. Clean text
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

# ==============================
# 3. Vectorization
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# ==============================
# 4. Model
# ==============================
model = PassiveAggressiveClassifier()
model.fit(X, df["label"])

# ==============================
# 5. Prediction
# ==============================
def predict_news(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred

# ==============================
# 6. Interactive
# ==============================
print("\n📰 Fake News Detector Ready\n")

while True:
    user_input = input("Enter news: ")

    if user_input.lower() == "exit":
        break

    result = predict_news(user_input)

    if result == 1:
        print("⚠️ FAKE NEWS\n")
    else:
        print("✅ REAL NEWS\n")