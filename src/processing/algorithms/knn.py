import time
import pandas as pd
import numpy as np
from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


class CustomKNN:
    def __init__(self, n_neighbors=5, weights="distance", p=2, algorithm="auto"):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
            algorithm=algorithm
        )
        self.distances_ = None
        self.indices_ = None
        self.X_train = None
        self.y_train = None
        self.vectorizer = None
        self.label_encoder = None
        self.original_docs = None

    def fit(self, X, y, original_docs=None, vectorizer=None, label_encoder=None):
        self.model.fit(X, y)
        self.X_train = X
        self.y_train = y
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.original_docs = original_docs

    def predict(self, X):
        distances, indices = self.model.kneighbors(X)
        self.distances_ = distances
        self.indices_ = indices

        predictions = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            labels = self.y_train[idx]
            label_weights = defaultdict(float)
            for label, d in zip(labels, dist):
                weight = 1 / d if d != 0 else 1e9
                label_weights[label] += weight

            max_weight = max(label_weights.values())
            top_labels = [lbl for lbl, wt in label_weights.items()
                          if wt == max_weight]

            if len(top_labels) == 1:
                predictions.append(top_labels[0])
            else:
                # === Custom tie-breaking ===
                tfidf_means = {}
                tfidf_totals = {}
                word_counts = {}
                doc_counts = {}
                for label in top_labels:
                    docs = [self.original_docs[j] for j in range(
                        len(self.y_train)) if self.y_train[j] == label]
                    tfidf_vectors = self.vectorizer.transform(docs)

                    tfidf_means[label] = tfidf_vectors.mean()
                    tfidf_totals[label] = tfidf_vectors.sum()
                    word_counts[label] = sum(len(doc.split()) for doc in docs)
                    doc_counts[label] = len(docs)

                max_mean = max(tfidf_means.values())
                top_mean = [
                    lbl for lbl in top_labels if tfidf_means[lbl] == max_mean]
                if len(top_mean) == 1:
                    predictions.append(top_mean[0])
                    continue

                max_total = max(tfidf_totals.values())
                top_total = [
                    lbl for lbl in top_mean if tfidf_totals[lbl] == max_total]
                if len(top_total) == 1:
                    predictions.append(top_total[0])
                    continue

                max_words = max(word_counts.values())
                top_words = [
                    lbl for lbl in top_total if word_counts[lbl] == max_words]
                if len(top_words) == 1:
                    predictions.append(top_words[0])
                    continue

                max_docs = max(doc_counts.values())
                top_docs = [
                    lbl for lbl in top_words if doc_counts[lbl] == max_docs]
                predictions.append(top_docs[0])

        return np.array(predictions)

    def score(self, X, y):
        return self.model.score(X, y)

    def get_distances(self):
        return self.distances_

    def get_indices(self):
        return self.indices_

    def get_neighbors_info(self, X, k=5):
        distances, indices = self.model.kneighbors(X, n_neighbors=k)
        neighbors_info = []

        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            neighbors = [{
                'index': idx,
                'label': self.y_train[idx],
                'text': self.original_docs[idx],
                'distance': dist
            } for idx, dist in zip(idxs, dists)]
            neighbors_info.append(neighbors)

        return neighbors_info


if __name__ == "__main__":
    max_features_options = [3500, 4000, 4250, 4750, 5000, None]
    test_size_options = [0.2, 0.25, 0.3]
    random_state_options = [4, 42, 100]
    n_neighbors_options = [3, 5, 7, 9, 11]

    dataset_path = "./src/storage/datasets/preprocessed/raw_news_dataset_preprocessed_stemmed.csv"
    df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")

    if df.empty:
        raise ValueError("Dataset kosong. Cek dataset Anda!")

    X_raw = df["preprocessedContent"].values
    y_raw = df["topik"].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    best_overall_accuracy = 0.0
    best_overall_config = None
    best_model = None

    start_time = time.time()
    print("Mulai evaluasi...")

    for max_feat, test_sz, rand_st in product(max_features_options, test_size_options, random_state_options):
        print(
            f"\n🔍 Evaluasi: max_features={max_feat}, test_size={test_sz}, random_state={rand_st}")

        tfidf_vectorizer = TfidfVectorizer(max_features=max_feat)
        X_tfidf = tfidf_vectorizer.fit_transform(X_raw)

        X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
            X_tfidf, y_encoded, X_raw, test_size=test_sz, stratify=y_encoded, random_state=rand_st
        )

        for n_neighbors in n_neighbors_options:
            knn = CustomKNN(n_neighbors=n_neighbors,
                            weights="distance", p=2, algorithm="auto")
            knn.fit(X_train, y_train, original_docs=raw_train,
                    vectorizer=tfidf_vectorizer, label_encoder=le)
            predictions = knn.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            print(
                f"    → Akurasi: {accuracy:.2%} | n_neighbors={n_neighbors}, weights=distance, p=2")

            if accuracy > best_overall_accuracy:
                best_overall_accuracy = accuracy
                best_overall_config = {
                    "max_features": max_feat,
                    "test_size": test_sz,
                    "random_state": rand_st,
                    "knn_params": {
                        "n_neighbors": n_neighbors,
                        "weights": "distance",
                        "p": 2
                    }
                }
                best_model = knn

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nWaktu eksekusi: {elapsed_time:.2f} detik")
    print("\n=== Model Terbaik Secara Keseluruhan ===")
    print(f"Konfigurasi: {best_overall_config}")
    print(f"Akurasi: {best_overall_accuracy:.2%}")
