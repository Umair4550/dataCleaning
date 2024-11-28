import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
def preprocess_text(text):
    """Clean and preprocess text data."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def load_and_preprocess_data(file_path):
    """Load, preprocess, and handle class imbalance."""
    data = pd.read_csv(file_path, usecols=["Questions", "Label"])
    print (data.head())
    data["Questions"] = data["Questions"].apply(preprocess_text)
    print (data.head())
    # Encode labels
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Label"])
    print (data.head())

    X = data["Questions"]
    y = data["Label"]
    undersample = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersample.fit_resample(X.values.reshape(-1, 1), y)
    X_resampled = pd.Series(X_resampled.flatten())
    y_resampled = pd.Series(y_resampled)

    return X_resampled, y_resampled, label_encoder

def train_tfidf_model(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with TF-IDF."""
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Transform the training and test data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_tfidf, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_tfidf)
    print(y_pred)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    print("\nTF-IDF + Logistic Regression Results:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return model, vectorizer


def predict_query(query, model, vectorizer=None, threshold=0.5):
    """Predict the class of a query using TF-IDF and add a threshold for irrelevance."""
    # Transform the query into TF-IDF features
    query_tfidf = vectorizer.transform([query])

    # Get prediction probabilities for all classes
    probas = model.predict_proba(query_tfidf)
    print(f"probas : {probas }")
    # Get the highest probability and the corresponding class
    max_proba = np.max(probas)
    print(f"probas : {probas}")
    predicted_class = np.argmax(probas)

    # If the confidence (max probability) is below the threshold, classify as "Other"
    if max_proba < threshold:
        return -1  # Return -1 for irrelevant queries
    else:
        return predicted_class  # Return the predicted class if confidence is above threshold


# Load and preprocess data
X, y, label_encoder = load_and_preprocess_data("Dataset/question.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
tfidf_model, tfidf_vectorizer = train_tfidf_model(X_train, y_train, X_test, y_test)

# Assuming the model and vectorizer are already trained
# Assuming the model and vectorizer are already trained
while True:
    query = input("\nEnter your query (type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Predict the query's class using TF-IDF model
    prediction = predict_query(query, tfidf_model, vectorizer=tfidf_vectorizer)

    # If prediction is -1 (irrelevant), print a message for irrelevant questions
    if prediction == -1:
        print("Query not related to admission or course.")
    else:
        # Decode the predicted class and print the result
        predicted_class = label_encoder.inverse_transform([prediction])
        print(f"Predicted Class: {predicted_class[0]}")
