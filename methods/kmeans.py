import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import time


def kmns(x_train, x_test, y_train, y_test, infoPrint):
    # K-MEANS CLASSIFIER
    if infoPrint:
        print("\n-- K-MEANS CLASSIFIER")

    kmeans_model = KMeans(n_clusters=3)
    kmeans_model.fit(x_train, y_train)
    y_pred_dct = kmeans_model.predict(x_test)
    kms_fpr, kms_tpr, thr = roc_curve(y_test, y_pred_dct)
    if infoPrint:
        print(classification_report(y_test, y_pred_dct))

    return kmeans_model, kms_fpr, kms_tpr


def kmns_k_search(x, genplot):
    print("-- Execute K-Means with Silhouette Score")
    tic = time.perf_counter()
    dataframe = pd.DataFrame(x)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(dataframe)
    scaled_dataframe = pd.DataFrame(scaled_array, columns=dataframe.columns)
    k_to_test = range(2, 100, 1)  # [2,3,4, ..., 24]
    silhouette_scores = {}
    score_k = []
    for k in k_to_test:
        model_kmeans_k = KMeans(n_clusters=k)
        model_kmeans_k.fit(scaled_dataframe)
        labels_k = model_kmeans_k.labels_
        score_k = metrics.silhouette_score(scaled_dataframe, labels_k)
        silhouette_scores[k] = score_k

    if genplot:
        plt.figure(figsize=(16, 5))
        plt.plot(silhouette_scores.values())
        plt.xticks(range(0, 98, 1), silhouette_scores.keys())
        plt.title("Silhouette Metric")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.axvline(1, color="r")
        # plt.show()
        plt.savefig("images/after_opt/BestK.png")
    toc = time.perf_counter()
    return min(silhouette_scores.values()), toc - tic
