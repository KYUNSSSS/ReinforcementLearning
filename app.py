import pandas as pd
import numpy as np
import random
import os
import pickle

# Load dataset
file_path = 'message.csv'
data = pd.read_csv(file_path)

# Initialize Q-table
q_table = {}
epsilon = 0.2  # Initial exploration rate

def load_q_table():
    global q_table
    if os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)

def save_q_table():
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

def initialize_q_table():
    global q_table
    for _, row in data.iterrows():
        q_table[(row['message'], row['persuasive_type'], row['activity'])] = 0  # Initialize Q-values to zero

# Function to get next message based on Q-values
def get_next_message():
    global epsilon
    if random.random() < epsilon:  # Exploration
        message = random.choice(list(q_table.keys()))
    else:  # Exploitation (choose best)
        message = max(q_table, key=q_table.get)
    
    epsilon = max(0.01, epsilon * 0.99)  # Gradually decrease exploration rate
    return message

# Save user responses
def save_responses(responses):
    save_path = 'user_responses2.csv'
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        new_df = pd.DataFrame(responses, columns=['message', 'persuasive_type', 'activity', 'response'])
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(responses, columns=['message', 'persuasive_type', 'activity', 'response'])
    
    df.to_csv(save_path, index=False)
    print(f"Responses saved to {save_path}")

# Reinforcement learning update
def update_q_table(message, persuasive_type, activity, reward, learning_rate=0.1, gamma=0.9):
    key = (message, persuasive_type, activity)
    previous_value = q_table.get(key, 0)
    
    # Reward shaping (increase reward if consistent positive feedback)
    if previous_value > 0 and reward == 1:
        reward += 0.2  # Encourage consistency
    
    q_table[key] = previous_value + learning_rate * (reward + gamma * max(q_table.values()) - previous_value)

# Main function
def run_reinforcement_learning():
    if not load_q_table():
        initialize_q_table()
        print("Answer with 1 (persuasive) or 0 (not persuasive). Press 'p' to stop.")
        
        # Initial 5 random messages
        responses = []
        initial_messages = random.sample(list(q_table.keys()), 5)
        for msg in initial_messages:
            persuasive_type, activity = msg[1], msg[2]
            print(f"Message: {msg[0]}\nPersuasive Type: {persuasive_type}\nActivity: {activity}")
            user_input = input("Enter 1 or 0: ")
            if user_input.lower() == 'p':
                save_responses(responses)
                save_q_table()
                return
            responses.append((msg[0], persuasive_type, activity, int(user_input)))
            update_q_table(msg[0], persuasive_type, activity, int(user_input))
        save_responses(responses)
        save_q_table()
    
    # Learning loop
    responses = []
    while True:
        msg = get_next_message()
        persuasive_type, activity = msg[1], msg[2]
        print(f"Message: {msg[0]}\nPersuasive Type: {persuasive_type}\nActivity: {activity}")
        user_input = input("Enter 1 or 0: ")
        if user_input.lower() == 'p':
            break
        responses.append((msg[0], persuasive_type, activity, int(user_input)))
        update_q_table(msg[0], persuasive_type, activity, int(user_input))
    
    save_responses(responses)
    save_q_table()
    print("Learning session completed.")

if __name__ == "__main__":
    run_reinforcement_learning()