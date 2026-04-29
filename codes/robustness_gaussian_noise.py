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


def evaluate(model, images, labels, is_cnn):
    if is_cnn:
        images = images.reshape(-1, 1, 28, 28)
    logits = model(images)
    return nn.metric.accuracy(logits, labels)


def main():
    np.random.seed(309)
    images, labels = load_test_data()
    sigmas = [0.0, 0.05, 0.10, 0.20, 0.30]

    mlp = nn.models.Model_MLP()
    mlp.load_model(os.path.join(BASE_DIR, 'best_models', 'mlp_run', 'best_model.pickle'))

    cnn = nn.models.Model_CNN()
    cnn.load_model(os.path.join(BASE_DIR, 'best_models', 'cnn_improved_run', 'best_model.pickle'))

    results = {'sigma': [], 'mlp': [], 'cnn': []}
    for sigma in sigmas:
        if sigma == 0:
            noisy = images.copy()
        else:
            noisy = np.clip(images + np.random.normal(0, sigma, images.shape), 0, 1)
        mlp_acc = evaluate(mlp, noisy, labels, is_cnn=False)
        cnn_acc = evaluate(cnn, noisy, labels, is_cnn=True)
        results['sigma'].append(sigma)
        results['mlp'].append(mlp_acc)
        results['cnn'].append(cnn_acc)
        print(f"sigma={sigma:.2f}, mlp_acc={mlp_acc:.4f}, cnn_acc={cnn_acc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(results['sigma'], results['mlp'], marker='o', label='MLP')
    plt.plot(results['sigma'], results['cnn'], marker='o', label='CNN')
    plt.xlabel('Gaussian noise sigma')
    plt.ylabel('Test accuracy')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'robustness_gaussian_noise.png')
    plt.savefig(out_path, dpi=200)
    print(f"saved figure to {out_path}")


if __name__ == '__main__':
    main()
