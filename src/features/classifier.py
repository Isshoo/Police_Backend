import time
import joblib
import pandas as pd
from src.preprocessing.extends.text_preprocessor import TextPreprocessor
from src.utilities.map_classification_result import map_classification_result


class TweetClassifier:
    def __init__(self, knn_model_path='./src/storage/models/base/knn_model.joblib'):
        """Inisialisasi model knn dan text preprocessor"""
        try:
            self.knn_model = joblib.load(knn_model_path)
        except Exception as e:
            print(f"❌ Gagal memuat knn model: {e}")
            self.knn_model = None  # Hindari crash jika model tidak bisa dimuat

        self.text_preprocessor = TextPreprocessor()
        self.valid_categories = {
            "Negatif", "Netral", "Positif"}  # Kategori yang valid

    def classify(self, sample_text):
        """ Mengklasifikasikan teks berita menggunakan model hybrid dan DeepSeek """
        processed_sample_text = self.text_preprocessor.preprocess(sample_text)

        try:
            hasil_model_knn = self.knn_model.predict(
                [processed_sample_text])[0]
            hasil_model_knn = map_classification_result(hasil_model_knn)
        except Exception as e:
            print(f"❌ Error pada model Hybrid: {e}")
            hasil_model_knn = "Unknown"

        return {
            "Preprocessed_Text": processed_sample_text,
            "KNN": hasil_model_knn,
        }

    def classify_csv(self, csv_file_path):
        """ Mengklasifikasikan CSV yang berisi berita """
        try:
            df = pd.read_csv(csv_file_path, encoding="utf-8",
                             on_bad_lines="skip")

            # Pastikan kolom yang diperlukan ada
            required_columns = {"komentar"}
            if not required_columns.issubset(df.columns):
                return {"error": f"File CSV harus memiliki kolom: {', '.join(required_columns)}"}

            df = df.dropna(subset=["komentar"])

            # Inisialisasi kolom hasil
            preprocessed_texts = []
            knn_results = []

            # Looping per baris untuk klasifikasi
            for text in df["contentSnippet"]:
                result = self.classify(text)
                preprocessed_texts.append(result["Preprocessed_Text"])
                knn_results.append(result["KNN"])

            # Tambahkan hasil ke dataframe
            df["Preprocessed_Text"] = preprocessed_texts
            df["KNN"] = knn_results

            return df.to_dict(orient="records")

        except pd.errors.EmptyDataError:
            return {"error": "File CSV kosong."}
        except pd.errors.ParserError as e:
            return {"error": f"Kesalahan parsing CSV: {str(e)}"}
        except UnicodeDecodeError:
            return {"error": "Encoding tidak valid. Coba simpan file sebagai UTF-8."}
        except Exception as e:
            return {"error": f"Kesalahan internal: {str(e)}"}
