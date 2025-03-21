import pandas as pd
from collections import defaultdict

def preprocess_data(data, min_dwell_time=15, max_dwell_time=1800, max_seq_length=30):
    """
    Preprocess user behavior data:
    - Remove abnormal dwell times
    - Convert data into trajectory sequences
    - Extract time slot features (morning, afternoon, evening)

    :param data: DataFrame containing user behavior data (user_id, timestamp, page_type, dwell_time)
    :param min_dwell_time: Minimum valid dwell time in seconds
    :param max_dwell_time: Maximum valid dwell time in seconds
    :param max_seq_length: Maximum trajectory length
    :return: Dictionary {user_id: list of (page_type, dwell_time)}
    """
    user_sessions = defaultdict(list)

    for user_id, group in data.groupby('user_id'):
        sequence = []
        for _, row in group.iterrows():
            if min_dwell_time <= row['dwell_time'] <= max_dwell_time:
                sequence.append((row['page_type'], row['dwell_time']))
        if len(sequence) > max_seq_length:
            sequence = sequence[:max_seq_length]
        if sequence:
            user_sessions[user_id] = sequence

    return user_sessions
