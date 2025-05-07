import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.knn_model import KNNClassifier
from src.utilities.model_evaluations import evaluate_model
from sklearn.preprocessing import LabelEncoder
from src.utilities.map_classification_result import map_classification_result
from src.utilities.save_model import save_model
from itertools import product
import time


class KNNModelTrainer:
    def __init__(self, dataset_path):
        """Inisialisasi trainer dengan dataset dalam bentuk DataFrame"""

        self.df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")

        if self.df.empty:
            raise ValueError("Dataset kosong. Cek dataset Anda!")

    def train(self, n_neighbors=11, test_size=0.25, max_features=None):
        """Melatih model Hybrid C5.0-KNN"""

        X_texts = self.df["preprocessedKomentar"].values
        y = self.df["label"].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
            X_texts, y_encoded, X_texts, test_size=test_size, stratify=y_encoded, random_state=42
        )

        knn_model = KNNClassifier(
            n_neighbors, max_features=max_features)

        # Latih model
        knn_model.fit(X_train, y_train, raw_train, le)

        tfidf_stats = knn_model.get_tfidf_word_stats(X_train)

        # Prediksi hasil
        y_pred = knn_model.predict(X_test)

        # Ambil vektor untuk X_test
        X_test_vectors = knn_model.vectorizer.transform(X_test)

        # Hitung tetangga terdekat untuk setiap data uji
        all_neighbors = []
        for i in range(len(X_test)):
            neighbors = knn_model.knn.get_neighbors_info(
                X_test_vectors[i], k=n_neighbors)[0]
            all_neighbors.append({
                "test_index": i,
                "test_text": raw_test[i],
                "predicted_label": le.inverse_transform([y_pred[i]])[0],
                "true_label": le.inverse_transform([y_test[i]])[0],
                "neighbors": neighbors
            })

        # Konversi ke DataFrame
        rows = []
        for item in all_neighbors:
            for neighbor in item["neighbors"]:
                rows.append({
                    "test_index": item["test_index"],
                    "test_text": item["test_text"],
                    "predicted_label": item["predicted_label"],
                    "true_label": item["true_label"],
                    "neighbor_index": neighbor["index"],
                    "neighbor_label": map_classification_result(neighbor["label"]),
                    "neighbor_distance": neighbor["distance"],
                    "neighbor_text": neighbor["text"]
                })

        df_neighbors = pd.DataFrame(rows)

        # Evaluasi model
        evaluation_results = evaluate_model(y_test, y_pred)

        return knn_model, evaluation_results, tfidf_stats, df_neighbors

    def train_with_gridsearch(self, param_grid=None):
        """Melatih model Hybrid C5.0-KNN dengan Grid Search untuk mencari parameter terbaik"""

        X_texts = self.df["preprocessedKomentar"].values
        y = self.df["label"].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Default grid search parameters
        if param_grid is None:
            param_grid = {
                "n_neighbors": [3, 5, 7, 9, 11],  # Coba beberapa nilai KNN
                # Coba beberapa split train-test
                "test_size": [0.2, 0.25, 0.3, 0.4],
                # Coba berbagai random state
                "random_state": [42, 100],
                # Coba beberapa nilai maksimum fitur
                "max_features": [None]
            }

        best_score = 0
        best_model = None
        best_params = None
        results = []

        # Loop melalui semua kombinasi parameter
        for n_neighbors, test_size, random_state, max_features in product(
            param_grid["n_neighbors"], param_grid["test_size"], param_grid["random_state"], param_grid["max_features"]
        ):
            print(
                f"🔍 Evaluating Hybrid Model with n_neighbors={n_neighbors}, test_size={test_size}, random_state={random_state}")

            print(f"📊 Data size before split: {len(X_texts)}")

            X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
                X_texts, y_encoded, X_texts, test_size=test_size, stratify=y_encoded, random_state=random_state
            )

            print(f"�� Train size: {len(X_train)}, Test size: {len(X_test)}")

            knn_model = KNNClassifier(
                n_neighbors=n_neighbors, max_features=max_features)
            # Latih model
            start_time = time.time()
            knn_model.fit(X_train, y_train, raw_train, le)
            train_duration = time.time() - start_time

            # Prediksi hasil
            y_pred = knn_model.predict(X_test)

            # Evaluasi model
            evaluation_results = evaluate_model(y_test, y_pred)
            accuracy = evaluation_results["accuracy"]

            print(f"✅ Accuracy: {accuracy:.4f}")

            results.append({
                "model": knn_model,
                "params": {
                    "n_neighbors": n_neighbors,
                    "test_size": test_size,
                    "random_state": random_state,
                    "train_duration": train_duration,
                    "max_features": max_features
                },
                "accuracy": accuracy
            })

            if accuracy > best_score:
                best_score = accuracy
                best_model = knn_model
                best_params = {
                    "n_neighbors": n_neighbors,
                    "test_size": test_size,
                    "random_state": random_state,
                    "train_duration": train_duration,
                    "max_features": max_features
                }

        # Urutkan hasil berdasarkan akurasi (tertinggi ke terendah)
        sorted_results = sorted(
            results, key=lambda x: x["accuracy"], reverse=True)

        print("\nRank\tAccuracy\tTest Size\tN_Neighbors\tRandom State\tTrain Duration\tMax Features")
        print(
            "----\t--------\t---------\t----------\t------------\t------------\t-------------")
        for i, result in enumerate(sorted_results, 1):
            accuracy = result["accuracy"]
            params = result["params"]
            print(
                f"{i}\t{accuracy:.4f}\t\t{params['test_size']}\t\t{params['n_neighbors']}\t\t{params['random_state']}\t\t{params['train_duration']:.2f}s\t\t{params['max_features']}")

        return best_model, best_params, best_score


if __name__ == "__main__":
    dataset_path = "./src/storage/datasets/preprocessed/datasetml2_original_preprocessed.csv"
    trainer = KNNModelTrainer(dataset_path)

    # # Training model secara manual
    # trainer.train(n_neighbors=11, test_size=0.25, max_features=None)

    # Training model dengan Grid Search
    best_model, best_params, best_score = trainer.train_with_gridsearch()
    print(f"\nParameter terbaik ditemukan: {best_params}")
    print(f"Akurasi terbaik: {best_score:.4f}")
    isSimpan = input("Apakah model akan disimpan sebagai default? y/n: ")
    if isSimpan.lower() == "y":
        save_model(best_model, "./src/storage/models/base/knn_model.joblib")
        print("Model hybrid disimpan sebagai default")
    else:
        print("Training Selesai")
