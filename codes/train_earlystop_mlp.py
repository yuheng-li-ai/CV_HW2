# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import os
import pickle

# fixed seed for experiment
np.random.seed(309)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_images_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(BASE_DIR, 'dataset', 'MNIST', 'train-labels-idx1-ubyte.gz')

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])

optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=None)

runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=5,
        log_iters=100,
        save_dir=os.path.join(BASE_DIR, 'best_models', 'earlystop_mlp_run'),
        early_stop=True,
        patience=1
)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

figure_path = os.path.join(BASE_DIR, 'figs', 'earlystop_mlp_curve.png')
plt.savefig(figure_path, dpi=200)
print(f"saved figure to {figure_path}")
plt.show()
