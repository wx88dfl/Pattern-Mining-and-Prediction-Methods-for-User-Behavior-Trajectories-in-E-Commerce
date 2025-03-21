import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

def find_frequent_trajectories(trajectories, min_support=2):
    """
    Perform frequent trajectory mining using the FP-Growth algorithm.

    :param trajectories: List of user trajectory sequences
    :param min_support: Minimum support threshold for frequent pattern mining
    :return: DataFrame containing frequent trajectory patterns
    """
    # Convert trajectories into a transactional format for FP-Growth
    unique_pages = set(sum(trajectories, []))
    transactions = pd.DataFrame([
        [1 if page in traj else 0 for page in unique_pages]
        for traj in trajectories
    ], columns=unique_pages)

    # Apply FP-Growth
    frequent_itemsets = fpgrowth(transactions, min_support=min_support/len(trajectories), use_colnames=True)

    return frequent_itemsets

if __name__ == "__main__":
    # Example trajectory data
    user_trajectories = [
        ['H', 'G', 'D'],
        ['H', 'G', 'F'],
        ['H', 'A', 'D'],
        ['H', 'G', 'D']
    ]

    # Perform frequent pattern mining
    frequent_patterns = find_frequent_trajectories(user_trajectories, min_support=2)

    # Print frequent patterns
    print(frequent_patterns)
