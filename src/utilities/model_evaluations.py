import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(y_test, y_pred):
    """Evaluasi model dengan confusion matrix dan classification report"""
    labels = np.unique(y_test)
    # akurasi
    report = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report_dict["accuracy"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return {
        "accuracy": report_dict["accuracy"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }
