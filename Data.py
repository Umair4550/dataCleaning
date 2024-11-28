import re
import csv

def convert_chat_to_qa(chat):
    qa_pairs = []
    lines = chat.strip().split('\n')

    for i in range(len(lines)):
        match = re.match(r'\[(.*?)\] (.*?): (.*)', lines[i])
        if match:
            timestamp, sender, message = match.groups()
            if i > 0:  # If it's not the first message
                previous_match = re.match(r'\[(.*?)\] (.*?): (.*)', lines[i - 1])
                if previous_match:
                    prev_timestamp, prev_sender, prev_message = previous_match.groups()
                    # Treat the previous message as the question and the current one as the answer
                    qa_pairs.append({'question': prev_message.strip(), 'answer': message.strip()})

    return qa_pairs

def save_to_csv(qa_pairs, filename='qa_dataset.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['question', 'answer'])
        writer.writeheader()
        for qa in qa_pairs:
            writer.writerow(qa)

# Read chat data from the _chat.txt file
with open('_chat.txt', 'r', encoding='utf-8') as file:
    chat_data = file.read()

# Convert chat data to Q&A pairs
qa_dataset = convert_chat_to_qa(chat_data)

# Save the Q&A pairs to a CSV file
save_to_csv(qa_dataset)

print(f"Q&A pairs saved to 'qa_dataset.csv'")
