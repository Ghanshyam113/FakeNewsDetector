import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine datasets
news = pd.concat([fake, true])

# Inputs and outputs
x = news["text"]
y = news["label"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=7
)

# Convert text into vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train, y_train)

# Test accuracy
prediction = model.predict(x_test)
score = accuracy_score(y_test, prediction)

print("Model Accuracy:", round(score*100,2), "%")

# User input
user_news = input("Enter News: ")

# Transform input
user_vector = vectorizer.transform([user_news])

# Predict
result = model.predict(user_vector)

print("Prediction:", result[0])