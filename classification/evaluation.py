import numpy as np
from pandas import DataFrame
from tensorflow import keras
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

def get_predictions_and_labels(model, dataset):
    raw_predictions = []
    predicted_labels = []
    true_labels = []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        raw_predictions.append(predictions)
        predicted_label = np.argmax(predictions, axis=1)
        predicted_labels.append(predicted_label)
        true_labels.append(labels.numpy())
    return np.concatenate(predicted_labels), np.concatenate(true_labels), np.concatenate(raw_predictions)

def get_accuracy(y_test, y_pred):
    return np.mean(y_pred == y_test)

def get_confusion_matrix(y_test, y_pred) -> np.ndarray:
    return confusion_matrix(y_test, y_pred)

def get_classification_report_pd(y_test, y_pred, class_names) -> dict:
    return DataFrame(classification_report(y_test, y_pred, target_names=class_names, output_dict=True))

def get_classification_report(y_test, y_pred, class_names) -> str:
    return classification_report(y_test, y_pred, target_names=class_names)