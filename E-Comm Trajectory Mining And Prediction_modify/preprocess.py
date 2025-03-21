import pandas as pd
from collections import defaultdict
from src.utils import log_message
from src.config import MIN_DWELL_TIME, MAX_DWELL_TIME, MAX_SEQUENCE_LENGTH

def preprocess_data(data):
    """
    Clean and preprocess user behavior data.

    :param data: DataFrame with user clickstream data
    :return: Dictionary {user_id: [(page_type, dwell_time)]}
    """
    user_sessions = defaultdict(list)

    for user_id, group in data.groupby('user_id'):
        sequence = [(row['page_type'], row['dwell_time'])
                    for _, row in group.iterrows()
                    if MIN_DWELL_TIME <= row['dwell_time'] <= MAX_DWELL_TIME]

        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]

        if sequence:
            user_sessions[user_id] = sequence

    log_message(f"Processed {len(user_sessions)} user sessions.")
    return user_sessions
