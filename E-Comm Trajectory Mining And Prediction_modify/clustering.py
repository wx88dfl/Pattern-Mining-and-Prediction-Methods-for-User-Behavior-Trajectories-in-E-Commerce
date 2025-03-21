import numpy as np
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from src.utils import log_message
from src.config import CLUSTERING_METHOD, DBSCAN_EPS, DBSCAN_MIN_SAMPLES

def dtw_distance(seq1, seq2):
    """
    Compute DTW (Dynamic Time Warping) distance between two trajectories.

    :param seq1: First trajectory sequence
    :param seq2: Second trajectory sequence
    :return: DTW distance
    """
    seq1, seq2 = [ord(x[0]) for x in seq1], [ord(x[0]) for x in seq2]
    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return distance

def cluster_trajectories(user_sessions):
    """
    Perform trajectory clustering using DBSCAN or HDBSCAN.

    :param user_sessions: Dictionary {user_id: trajectory sequence}
    :return: Dictionary {user_id: cluster label}
    """
    trajectories = list(user_sessions.values())
    distance_matrix = np.array([[dtw_distance(seq1, seq2) for seq2 in trajectories] for seq1 in trajectories])

    if CLUSTERING_METHOD == "dbscan":
        clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed").fit(distance_matrix)
    else:
        clustering = HDBSCAN(min_cluster_size=5, metric="precomputed").fit(distance_matrix)

    log_message(f"Clustered {len(user_sessions)} user trajectories using {CLUSTERING_METHOD}.")
    return {user_id: label for user_id, label in zip(user_sessions.keys(), clustering.labels_)}
