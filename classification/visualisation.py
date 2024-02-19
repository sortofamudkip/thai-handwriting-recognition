import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from tensorflow import keras


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: list, save_path: Path | None = None):
    plt.figure(figsize=(20,20))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)

def plot_training_history(history: keras.callbacks.History, save_path: Path | None = None):
    plt.figure()
    plt.plot(history.history['sparse_categorical_accuracy'], label='train accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training History')
    if save_path:
        plt.savefig(save_path)

def plot_loss_history(history: keras.callbacks.History, save_path: Path | None = None):
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    if save_path:
        plt.savefig(save_path)


def plot_sample_predictions(model: keras.Model, dataset: keras.utils.Sequence, class_names: list, save_path: Path | None = None):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for images, labels in dataset.take(1):
        predictions = model.predict(images, verbose=0)
        for i in range(9):
            ax = axes[i // 3, i % 3]
            ax.imshow(images[i].numpy().reshape(64, 64), cmap='gray')
            ax.set_title(f'Predicted: {class_names[np.argmax(predictions[i])]}\nTrue: {class_names[labels[i]]}')
            ax.axis('off')
    if save_path:
        plt.savefig(save_path)

