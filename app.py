import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import pickle

# Load dataset
file_path = 'message.csv'
data = pd.read_csv(file_path)

# Initialize Q-table
if 'q_table' not in st.session_state:
    st.session_state.q_table = {}
if 'epsilon' not in st.session_state:
    st.session_state.epsilon = 0.2  # Initial exploration rate

def load_q_table():
    if os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as f:
            st.session_state.q_table = pickle.load(f)
        return True
    return False

def save_q_table():
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(st.session_state.q_table, f)

def initialize_q_table():
    for _, row in data.iterrows():
        st.session_state.q_table[(row['message'], row['persuasive_type'], row['activity'])] = 0  # Initialize Q-values to zero

def get_next_message():
    if random.random() < st.session_state.epsilon:  # Exploration
        message = random.choice(list(st.session_state.q_table.keys()))
    else:  # Exploitation (choose best)
        message = max(st.session_state.q_table, key=st.session_state.q_table.get)
    
    st.session_state.epsilon = max(0.01, st.session_state.epsilon * 0.99)  # Gradually decrease exploration rate
    return message

def save_responses(responses):
    save_path = 'user_responses.csv'
    new_df = pd.DataFrame(responses, columns=['message', 'persuasive_type', 'activity', 'response'])
    new_df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
    st.write(f"Responses appended to {save_path}")

def update_q_table(message, persuasive_type, activity, reward, learning_rate=0.1, gamma=0.9):
    key = (message, persuasive_type, activity)
    previous_value = st.session_state.q_table.get(key, 0)
    
    # Reward shaping (increase reward if consistent positive feedback)
    if previous_value > 0 and reward == 1:
        reward += 0.2  # Encourage consistency
    
    st.session_state.q_table[key] = previous_value + learning_rate * (reward + gamma * max(st.session_state.q_table.values()) - previous_value)

def main():
    if not load_q_table():
        initialize_q_table()
        st.write("Answer with 1 (persuasive) or 0 (not persuasive). Press 'p' to stop.")
        
        # Initial 5 random messages
        responses = []
        initial_messages = random.sample(list(st.session_state.q_table.keys()), 5)
        for msg in initial_messages:
            persuasive_type, activity = msg[1], msg[2]
            st.write(f"Message: {msg[0]}\nPersuasive Type: {persuasive_type}\nActivity: {activity}")
            user_input = st.radio("Is this persuasive?", options=[1, 0], key=f"response_{msg[0]}")
            responses.append((msg[0], persuasive_type, activity, int(user_input)))
            update_q_table(msg[0], persuasive_type, activity, int(user_input))
        save_responses(responses)
        save_q_table()
    
    # Learning loop
    responses = []
    msg = get_next_message()
    persuasive_type, activity = msg[1], msg[2]
    st.write(f"Message: {msg[0]}\n\nPersuasive Type: {persuasive_type}\n\nActivity: {activity}")
    user_input = st.radio("Is this persuasive?", options=[1, 0], key=f"response_{msg[0]}")
    if st.button('Submit'):
        responses.append((msg[0], persuasive_type, activity, int(user_input)))
        update_q_table(msg[0], persuasive_type, activity, int(user_input))
        save_responses(responses)
        save_q_table()
        st.rerun()

    if os.path.exists('user_responses.csv'):
        with open('user_responses.csv', 'rb') as f:
            csv_data = f.read()
        st.download_button(
            label="Download User Responses",
            data=csv_data,
            file_name="user_responses.csv",
            mime="text/csv"
        )



if __name__ == "__main__":
    main()