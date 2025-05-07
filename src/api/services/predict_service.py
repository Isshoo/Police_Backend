from src.features.classifier import TweetClassifier


class PredictService:
    PREDICT_DIR = "src/storage/datasets/classifications"

    def predict(self, text, model_path='./src/storage/models/base/knn_model.joblib'):
        """ Mengklasifikasikan teks berita menggunakan model hybrid dan DeepSeek """
        classifier = TweetClassifier(model_path)
        result = classifier.classify(text)
        return result

    def predict_csv(self, csv_file_path, model_path='./src/storage/models/base/knn_model.joblib'):
        """ Mengklasifikasikan teks berita dari file CSV """
        classifier = TweetClassifier(model_path)
        result = classifier.classify_csv(csv_file_path)
        return result
