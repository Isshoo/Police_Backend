import joblib


def save_model(model, model_path):
    """Menyimpan model dalam format joblib"""
    joblib.dump(model, model_path)
    print(f"Model berhasil disimpan di {model_path}")
