from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import re

# Load the CSV file
data = pd.read_csv("datasetrulebased.csv")

# Extract questions, answers, and categories
questions = data['Question'].tolist()
answers = data['Answer'].tolist()
categories = data['Label'].tolist()  # Assuming you have a Category column in your CSV

# Load the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight SBERT model

# Encode the questions
question_embeddings = model.encode(questions, convert_to_tensor=True)


# Function to preprocess text (lowercasing, stripping extra spaces, removing punctuation)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def generate_response(user_query):
    # Preprocess the user query
    user_query = preprocess_text(user_query)

    # Encode the user query
    user_query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(user_query_embedding, question_embeddings)

    # Get the most similar question index
    most_similar_idx = np.argmax(similarities).item()
    similarity_score = similarities[0][most_similar_idx].item()

    # Return the answer if similarity is above a threshold
    if similarity_score > 0.6:  # Adjusted threshold to improve accuracy
        response = answers[most_similar_idx]
        category = categories[most_similar_idx]  # Fetch category
        return response, category, similarity_score
    else:
        return "I'm sorry, I couldn't find a relevant answer. Please rephrase your question.", None, similarity_score


def calculate_total_accuracy():
    correct_count = 0
    total_count = len(questions)

    for i, question in enumerate(questions):
        response, _, similarity_score = generate_response(question)
        if response == answers[i]:  # Comparing response with the actual answer
            correct_count += 1

    accuracy = (correct_count / total_count) * 100
    return accuracy


# Test the system
if __name__ == "__main__":
    print("Chatbot is ready! Type your question below:")

    # Calculate and show total model accuracy
    total_accuracy = calculate_total_accuracy()
    print(f"Total Model Accuracy: {total_accuracy:.2f}%\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break
        response, category, similarity_score = generate_response(user_input)

        # Show the response, category, and accuracy
        if category:
            print(f"Chatbot: {response}")
            print(f"Category: {category}")
            print(f"Accuracy: {similarity_score:.2f}")
        else:
            print(f"Chatbot: {response}")
            print(f"Accuracy: {similarity_score:.2f}")
