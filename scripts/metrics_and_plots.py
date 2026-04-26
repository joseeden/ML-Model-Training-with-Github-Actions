import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

OUTPUTS_DIR = "outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)


def plot_confusion_matrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap=plt.cm.Blues)
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))


def save_metrics(metrics):
    with open(os.path.join(OUTPUTS_DIR, "metrics.json"), "w") as fp:
        json.dump(metrics, fp)
