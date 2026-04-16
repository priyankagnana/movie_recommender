# sentiment_analyzer.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# 1. Dataset
# ==============================
data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Very happy with the service",
        "Absolutely fantastic experience",
        "I hate this",
        "Very bad experience",
        "Not satisfied",
        "Worst purchase ever",
        "I am so happy",
        "This is terrible"
    ],
    "label": [1,1,1,1,0,0,0,0,1,0]  # 1 = positive, 0 = negative
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
# 3. Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ==============================
# 4. Vectorization
# ==============================
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 5. Model
# ==============================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ==============================
# 6. Evaluation
# ==============================
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ==============================
# 7. Prediction function
# ==============================
def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    return pred, prob

# ==============================
# 8. User interaction
# ==============================
print("\n💬 Sentiment Analyzer Ready (type 'exit' to quit)\n")

while True:
    user_input = input("Enter text: ")

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break

    pred, prob = predict_sentiment(user_input)

    if pred == 1:
        print(f"😊 POSITIVE (Confidence: {max(prob)*100:.2f}%)\n")
    else:
        print(f"😡 NEGATIVE (Confidence: {max(prob)*100:.2f}%)\n")
        