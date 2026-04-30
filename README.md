📧 Email & SMS Spam Classification

A Machine Learning project that classifies messages as Spam or Not Spam (Ham) using Natural Language Processing (NLP) techniques and supervised learning algorithms.

🚀 Project Overview

Spam messages are a common problem in emails and SMS. This project uses text preprocessing, TF-IDF vectorization, and machine learning models to accurately detect spam messages.

The model is trained on labeled data and can predict whether a given message is spam or not.

🧠 Features
Text preprocessing (lowercasing, tokenization, stopword removal)
TF-IDF vectorization
Multiple ML models comparison
Voting Classifier for better accuracy
Model & vectorizer saved using pickle
Streamlit web app for user interaction
🛠️ Tech Stack
Programming Language: Python
Libraries:
NumPy
Pandas
Scikit-learn
NLTK
Streamlit
ML Algorithms:
Multinomial Naive Bayes
Logistic Regression
Support Vector Machine (SVM)
📂 Project Structure
email-sms-spam-classifier/
│
├── app.py                  # Streamlit app
├── train.py                # Model training script
├── model.pkl               # Trained ML model
├── vectorizer.pkl          # TF-IDF vectorizer
├── spam.csv                # Dataset
├── requirements.txt        # Required libraries
└── README.md               # Project documentation
📊 Dataset
Public spam dataset containing labeled spam and ham messages.
Each row contains:
Message text
Label (spam / ham)
⚙️ How It Works
Input Message
Text Preprocessing
TF-IDF Vectorization
Model Prediction
Spam / Not Spam Output
▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Ravi-tech17/email-sms-spam-classifier.git
cd email-sms-spam-classifier
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py
🧪 Model Performance
Accuracy: ~97%
Precision (Spam): High
Voting Classifier used for improved results
💾 Saving the Model
import pickle

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
