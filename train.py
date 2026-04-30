import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import urllib.request
import zipfile
import os

print("🔥 TRAINING NEW MODEL...")

# Download dataset
print("📥 Downloading SMS Spam Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
urllib.request.urlretrieve(url, "smsspamcollection.zip")

with zipfile.ZipFile("smsspamcollection.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
df.to_csv('spam.csv', index=False)
os.remove("smsspamcollection.zip")
print("✅ Dataset ready! Shape:", df.shape)

# Clean data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()
print(f"📊 Dataset: {len(df)} messages")

# Split data
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🤖 Training...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)  # ✅ THIS FITS THE MODEL

# Test
X_test_tfidf = tfidf.transform(X_test)
accuracy = model.score(X_test_tfidf, y_test)
print(f"✅ ACCURACY: {accuracy:.3f}")

# Save NEW files
print("💾 Saving...")
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("🎉 SUCCESS!")
print("Files created:")
print("   ✅ vectorizer.pkl")
print("   ✅ model.pkl")
print("   ✅ spam.csv")