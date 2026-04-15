# spam_classifier_full.py

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. Sample Dataset
# ==============================
data = {
    "text": [
        "Win money now",
        "Limited offer just for you",
        "Earn cash fast",
        "Congratulations you won a lottery",
        "Free entry in contest",
        "Meeting at 10 am",
        "Project deadline tomorrow",
        "Let's discuss work",
        "Can you review my code?",
        "Lunch at 1?"
    ],
    "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 = spam, 0 = ham
}

df = pd.DataFrame(data)

# ==============================
# 2. Text Cleaning Function
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

# ==============================
# 3. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ==============================
# 4. Feature Extraction (TF-IDF)
# ==============================
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 5. Model Training
# ==============================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ==============================
# 6. Evaluation
# ==============================
y_pred = model.predict(X_test_vec)

print("\n📊 Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 7. User Input Prediction
# ==============================
def predict_spam(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    return result, prob

# ==============================
# 8. Interactive Mode
# ==============================
print("\n🤖 Spam Classifier Ready (type 'exit' to quit)\n")

while True:
    user_input = input("Enter message: ")

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break

    result, prob = predict_spam(user_input)

    if result == 1:
        print(f"🚫 SPAM (Confidence: {max(prob)*100:.2f}%)\n")
    else:
        print(f"✅ NOT SPAM (Confidence: {max(prob)*100:.2f}%)\n")