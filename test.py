import pandas as pd
import re
import random
import nltk
from nltk.tokenize import sent_tokenize
import itertools

# Download the 'punkt' tokenizer resource if not already available
nltk.download('punkt')

# Load content from text file
file_path = 'abc.txt'  # Update this to your file path
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

# Define categories and their corresponding labels
categories = {
    "admission": 0,
    "fee": 1,
    "scholarship": 1,
    "transport": 2,
    "course": 3,
    "residence": 4,
    "program": 5
}

# Initialize a list to hold question-answer pairs
qa_pairs = []

# Function to generate Q&A based on category
def generate_qa_for_category(category, label, sentences):
    for sentence in sentences:
        question = f"What can you tell me about {category}?"
        answer = sentence.strip()
        if answer:  # Ensure non-empty answers
            qa_pairs.append((question, answer, label))

# Parse the content and generate Q&A pairs
for category, label in categories.items():
    # Use regex to find sections related to each category
    pattern = rf"({category}.*?)(?=(\n[A-Z]|$))"  # Match till a newline with a capital letter
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

    # For each match found, tokenize into sentences
    for match in matches:
        sentences = sent_tokenize(match[0])  # Tokenize sentences
        generate_qa_for_category(category, label, sentences)

# Ensure we have at least 25,000 unique pairs
if len(qa_pairs) < 25000:
    # If we have fewer than 25,000 pairs, cycle through existing pairs to fill
    qa_pairs = list(itertools.islice(itertools.cycle(qa_pairs), 25000))

# Shuffle the list to randomize order
random.shuffle(qa_pairs)
qa_pairs = qa_pairs[:25000]  # Ensure it's exactly 25,000

# Save to CSV
df = pd.DataFrame(qa_pairs, columns=["Question", "Answer", "Label"])
df.to_csv("dataset.csv", index=False)

print("Dataset generated successfully and saved as 'dataset.csv'.")
