import os
import gzip
from struct import unpack

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)


def load_test_data():
    images_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-images-idx3-ubyte.gz')
    labels_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 't10k-labels-idx1-ubyte.gz')
    with gzip.open(images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28) / 255.0
    with gzip.open(labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


def predict(model, images, is_cnn):
    if is_cnn:
        images = images.reshape(-1, 1, 28, 28)
    logits = model(images)
    return np.argmax(logits, axis=1)


def save_confusion_matrix(labels, preds, name):
    matrix = np.zeros((10, 10), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[label, pred] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f'{name}_confusion_matrix.png')
    plt.savefig(out_path, dpi=200)
    print(f"saved figure to {out_path}")
    return matrix


def save_misclassified(images, labels, preds, name, count=16):
    wrong = np.where(labels != preds)[0][:count]
    rows, cols = 4, 4
    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(wrong):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"T:{labels[idx]} P:{preds[idx]}")
        ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f'{name}_misclassified.png')
    plt.savefig(out_path, dpi=200)
    print(f"saved figure to {out_path}")


def save_mlp_weights(model):
    weights = model.layers[0].params['W'].T[:64]
    plt.figure(figsize=(8, 8))
    for i, w in enumerate(weights):
        ax = plt.subplot(8, 8, i + 1)
        ax.imshow(w.reshape(28, 28), cmap='coolwarm')
        ax.axis('off')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'mlp_first_layer_weights.png')
    plt.savefig(out_path, dpi=200)
    print(f"saved figure to {out_path}")


def save_cnn_kernels(model):
    kernels = model.layers[0].params['W'][0]
    count = kernels.shape[0]
    plt.figure(figsize=(2 * count, 2))
    for i in range(count):
        ax = plt.subplot(1, count, i + 1)
        ax.imshow(kernels[i, 0], cmap='coolwarm')
        ax.axis('off')
        ax.set_title(f"K{i}")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'cnn_first_layer_kernels.png')
    plt.savefig(out_path, dpi=200)
    print(f"saved figure to {out_path}")


def main():
    images, labels = load_test_data()

    mlp = nn.models.Model_MLP()
    mlp.load_model(os.path.join(BASE_DIR, 'best_models', 'mlp_run', 'best_model.pickle'))
    mlp_preds = predict(mlp, images, is_cnn=False)
    print(f"mlp_acc={np.mean(mlp_preds == labels):.4f}")
    save_confusion_matrix(labels, mlp_preds, 'mlp')
    save_misclassified(images, labels, mlp_preds, 'mlp')
    save_mlp_weights(mlp)

    cnn = nn.models.Model_CNN()
    cnn.load_model(os.path.join(BASE_DIR, 'best_models', 'cnn_improved_run', 'best_model.pickle'))
    cnn_preds = predict(cnn, images, is_cnn=True)
    print(f"cnn_acc={np.mean(cnn_preds == labels):.4f}")
    save_confusion_matrix(labels, cnn_preds, 'cnn')
    save_misclassified(images, labels, cnn_preds, 'cnn')
    save_cnn_kernels(cnn)


if __name__ == '__main__':
    main()
