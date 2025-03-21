import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from src.utils import log_message
from src.config import FP_GROWTH_MIN_SUPPORT

def find_frequent_trajectories(trajectories):
    """
    Perform frequent trajectory mining using the FP-Growth algorithm.

    :param trajectories: List of user trajectory sequences
    :return: DataFrame containing frequent trajectory patterns
    """
    unique_pages = set(sum(trajectories, []))
    transactions = pd.DataFrame([
        [1 if page in traj else 0 for page in unique_pages]
        for traj in trajectories
    ], columns=unique_pages)

    frequent_itemsets = fpgrowth(transactions, min_support=FP_GROWTH_MIN_SUPPORT, use_colnames=True)

    log_message(f"Found {len(frequent_itemsets)} frequent trajectory patterns.")
    return frequent_itemsets
