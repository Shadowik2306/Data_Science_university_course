import math
from typing import Dict, List, Tuple

import numpy as np
from pandas import DataFrame
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score


def run_agglomerative(
    df: DataFrame, num_clusters: int | None = 2
) -> cluster.AgglomerativeClustering:
    agglomerative = cluster.AgglomerativeClustering(
        n_clusters=num_clusters,
        compute_distances=True,
    )
    return agglomerative.fit(df)


def get_linkage_matrix(model: cluster.AgglomerativeClustering) -> np.ndarray:
    counts = np.zeros(model.children_.shape[0])  # type: ignore
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):  # type: ignore
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack([model.children_, model.distances_, counts]).astype(float)


def print_cluster_result(
    df: DataFrame, clusters_num: int, labels: np.ndarray, separator: str = ", "
):
    for cluster_id in range(clusters_num):
        cluster_indices = np.where(labels == cluster_id)[0]
        print(f"Cluster {cluster_id + 1} ({len(cluster_indices)}):")
        rules = [str(df.index[idx]) for idx in cluster_indices]
        print(separator.join(rules))
        print("")
        print("--------")


def run_kmeans(
    df: DataFrame, num_clusters: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(df)
    return labels, kmeans.cluster_centers_


def fit_kmeans(
    reduced_data: np.ndarray, num_clusters: int, random_state: int
) -> cluster.KMeans:
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=random_state)
    kmeans.fit(reduced_data)
    return kmeans


def _get_kmeans_range(
    df: DataFrame | np.ndarray, random_state: int
) -> Tuple[List, range]:
    max_clusters = int(math.sqrt(len(df)))
    clusters_range = range(2, max_clusters + 1)
    kmeans_per_k = [
        cluster.KMeans(n_clusters=k, random_state=random_state).fit(df)
        for k in clusters_range
    ]
    return kmeans_per_k, clusters_range


def get_clusters_inertia(df: DataFrame, random_state: int) -> Tuple[List, range]:
    kmeans_per_k, clusters_range = _get_kmeans_range(df, random_state)
    return [model.inertia_ for model in kmeans_per_k], clusters_range


def get_clusters_silhouette_scores(
    df: DataFrame, random_state: int
) -> Tuple[List, range]:
    kmeans_per_k, clusters_range = _get_kmeans_range(df, random_state)
    return [
        float(silhouette_score(df, model.labels_)) for model in kmeans_per_k
    ], clusters_range


def get_clusters_silhouettes(df: np.ndarray, random_state: int) -> Dict:
    kmeans_per_k, _ = _get_kmeans_range(df, random_state)
    clusters_silhouettes: Dict = {}
    for model in kmeans_per_k:
        silhouette_value = silhouette_score(df, model.labels_)
        sample_silhouette_values = silhouette_samples(df, model.labels_)
        clusters_silhouettes[model.n_clusters] = (
            silhouette_value,
            sample_silhouette_values,
            model,
        )
    return clusters_silhouettes
