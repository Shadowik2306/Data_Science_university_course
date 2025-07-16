from typing import Any, Dict, List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans


def draw_data_2d(
    df: DataFrame,
    col1: int,
    col2: int,
    y: List | None = None,
    classes: List | None = None,
    subplot: Any | None = None,
):
    ax = None
    if subplot is None:
        _, ax = plt.subplots()
    else:
        ax = subplot
    scatter = ax.scatter(df[df.columns[col1]], df[df.columns[col2]], c=y)
    ax.set(xlabel=df.columns[col1], ylabel=df.columns[col2])
    if classes is not None:
        ax.legend(
            scatter.legend_elements()[0], classes, loc="lower right", title="Classes"
        )


def draw_dendrogram(linkage_matrix: np.ndarray):
    hierarchy.dendrogram(linkage_matrix, truncate_mode="level", p=3)


def draw_cluster_results(
    df: DataFrame,
    col1: int,
    col2: int,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    subplot: Any | None = None,
):
    ax = None
    if subplot is None:
        ax = plt
    else:
        ax = subplot

    centroids = cluster_centers
    u_labels = np.unique(labels)

    for i in u_labels:
        ax.scatter(
            df[labels == i][df.columns[col1]],
            df[labels == i][df.columns[col2]],
            label=i,
        )

    ax.scatter(centroids[:, col1], centroids[:, col2], s=80, color="k")


def draw_clusters(reduced_data: np.ndarray, kmeans: KMeans):
    h = 0.02

    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,  # type: ignore
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())


def _draw_cluster_scores(
    data: List,
    clusters_range: range,
    score_name: str,
    title: str,
):
    plt.figure(figsize=(8, 5))
    plt.plot(clusters_range, data, "bo-")
    plt.xlabel("$k$", fontsize=8)
    plt.ylabel(score_name, fontsize=8)
    plt.title(title)


def draw_elbow_diagram(inertias: List, clusters_range: range):
    _draw_cluster_scores(inertias, clusters_range, "Inertia", "The Elbow Diagram")


def draw_silhouettes_diagram(silhouette: List, clusters_range: range):
    _draw_cluster_scores(
        silhouette, clusters_range, "Silhouette score", "The Silhouette score"
    )


def _draw_silhouette(
    ax: Any,
    reduced_data: np.ndarray,
    n_clusters: int,
    silhouette_avg: float,
    sample_silhouette_values: List,
    cluster_labels: List,
):
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(reduced_data) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)  # type: ignore
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


def _draw_cluster_data(
    ax: Any,
    reduced_data: np.ndarray,
    n_clusters: int,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
):
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)  # type: ignore
    ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        marker=".",
        s=30,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    ax.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(cluster_centers):
        ax.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")


def draw_silhouettes(reduced_data: np.ndarray, silhouettes: Dict):
    for key, value in silhouettes.items():
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        n_clusters = key
        silhouette_avg = value[0]
        sample_silhouette_values = value[1]
        cluster_labels = value[2].labels_
        cluster_centers = value[2].cluster_centers_

        _draw_silhouette(
            ax1,
            reduced_data,
            n_clusters,
            silhouette_avg,
            sample_silhouette_values,
            cluster_labels,
        )

        _draw_cluster_data(
            ax2,
            reduced_data,
            n_clusters,
            cluster_labels,
            cluster_centers,
        )

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
