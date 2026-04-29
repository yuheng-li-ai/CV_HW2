# Project 1 Final Report

**Title:** Project 1 of Neural Network and Deep Learning

**Name:** Yuheng Li

**Student ID:** 23307130334

**Github Repository:** https://github.com/yuheng-li-ai/CV_HW2

**Model Weights / Checkpoints:** [Insert Link]

## 1. MLP Baseline

This project studies handwritten digit classification on the MNIST dataset. I first implemented and trained a Multi-Layer Perceptron as the baseline model. The MLP baseline is important because it gives a simple reference point before introducing convolutional structure.

For the MLP implementation, I completed the forward and backward propagation of the linear layer and implemented the cross-entropy loss with softmax. The input image was flattened from a `28 x 28` image into a 784-dimensional vector. The model structure was `784-600-10` with ReLU activation. The training setting used learning rate `0.06`, batch size `32`, and `5` epochs.

The MLP baseline converged stably. The best validation accuracy recorded during training was `0.86660`, and the final test accuracy of the saved MLP checkpoint was `0.8724`. The learning curve is saved as `codes/figs/mlp_curve.png`.

## 2. CNN Model and MLP-vs-CNN Comparison

After building the MLP baseline, I implemented a simple CNN for the same MNIST classification task. The convolution operator was implemented manually in NumPy rather than using deep learning library operators. The final CNN used one convolution layer, ReLU activation, flattening, one hidden linear layer, another ReLU activation, and a final linear classifier.

The CNN setting was kept close to the MLP training setting: learning rate `0.06`, batch size `32`, and `5` epochs. The final CNN structure used `8` convolution channels, kernel size `3`, and hidden dimension `128`. This remains a simple CNN, without bottleneck blocks, residual connections, or other advanced architecture design.

The final CNN achieved best validation accuracy `0.94430` and test accuracy `0.9467`. This is clearly better than the MLP baseline test accuracy `0.8724`. The result supports the expected observation that CNN is more suitable for image classification because convolution uses local receptive fields and shared filters, while the MLP treats the image as a flattened vector and loses explicit spatial structure.

The CNN learning curve is saved as `codes/figs/cnn_improved_curve.png`.

## 3. Additional Directions

The PDF requires two additional directions after the MLP and CNN baselines. I selected optimization and regularization as the main two directions. I also conducted robustness analysis and error analysis/visualization because they directly support the discussion and detailed visualization requirements.

### 3.1 Optimization: Momentum

For the optimization direction, I implemented momentum gradient descent. Momentum was applied separately to the MLP and CNN models. The comparison baseline is the corresponding model trained with plain SGD under the baseline setting.

For MLP, momentum improved the test accuracy from `0.8724` to `0.9389`. This shows that momentum helped the MLP optimize more effectively under the current setting.

For CNN, momentum improved the test accuracy from `0.9467` to `0.9758`. The best validation accuracy of the CNN momentum run was `0.97520`. This is the strongest result among the experiments. The CNN momentum curve is saved as `codes/figs/opt_momentum_cnn_improved_curve.png`, and the MLP momentum curve is saved as `codes/figs/opt_momentum_curve.png`.

The conclusion is that momentum is useful for both models, especially for the CNN. It accelerates and stabilizes optimization by accumulating a velocity term instead of using only the current gradient.

### 3.2 Regularization: Early Stopping

For the regularization direction, I tested early stopping. The early stopping rule monitors validation accuracy at epoch boundaries and stops training if the validation score does not improve for the patience setting.

For MLP, early stopping did not activate during the 5-epoch run. The final test accuracy was `0.8724`, the same as the MLP baseline. This means early stopping had no practical effect under this setting because the validation accuracy kept improving until the normal end of training.

For CNN, early stopping also did not activate in the final improved run. The best validation accuracy was `0.94430`, and the test accuracy was `0.9467`, matching the improved CNN baseline. This suggests that early stopping is not informative in this experiment setting because the model did not show validation degradation within 5 epochs.

The early stopping curves are saved as `codes/figs/earlystop_mlp_curve.png` and `codes/figs/earlystop_cnn_improved_curve.png`.

### 3.3 Robustness Analysis: Gaussian Noise

For robustness analysis, I evaluated trained MLP and CNN models under Gaussian noise added to the test images. No retraining was used. The purpose was to examine model stability under input perturbation rather than clean accuracy only.

| Gaussian noise sigma | MLP accuracy | CNN accuracy |
| --- | --- | --- |
| 0.00 | 0.8724 | 0.9467 |
| 0.05 | 0.8621 | 0.9470 |
| 0.10 | 0.8212 | 0.9437 |
| 0.20 | 0.6789 | 0.9152 |
| 0.30 | 0.5434 | 0.8153 |

The CNN remained more stable than the MLP as noise increased. At sigma `0.30`, the MLP dropped to `0.5434`, while the CNN still reached `0.8153`. This suggests that the CNN learned more robust image features than the MLP baseline.

The robustness figure is saved as `codes/figs/robustness_gaussian_noise.png`.

### 3.4 Error Analysis and Visualization

I also analyzed model errors and learned representations through confusion matrices, misclassified samples, MLP weights, and CNN kernels. These visualizations help explain what the models learned and what types of samples remain difficult.

The MLP and CNN confusion matrices are saved as `codes/figs/mlp_confusion_matrix.png` and `codes/figs/cnn_confusion_matrix.png`. Misclassified examples are saved as `codes/figs/mlp_misclassified.png` and `codes/figs/cnn_misclassified.png`. The MLP first-layer weight visualization is saved as `codes/figs/mlp_first_layer_weights.png`, and the CNN first-layer kernels are saved as `codes/figs/cnn_first_layer_kernels.png`.

The error analysis confirms the quantitative results: the CNN makes fewer mistakes than the MLP and learns spatial filters that are more suitable for image inputs.

## 4. Main Results Table

| Experiment | Validation accuracy | Test accuracy | Main observation |
| --- | --- | --- | --- |
| MLP baseline | 0.86660 | 0.8724 | Stable baseline, but limited by flattened input representation |
| CNN baseline | 0.94430 | 0.9467 | CNN clearly improves over MLP |
| MLP + Momentum | [Not recorded in log] | 0.9389 | Momentum improves MLP optimization |
| CNN + Momentum | 0.97520 | 0.9758 | Best overall result |
| MLP + Early stopping | 0.86660 | 0.8724 | Early stopping did not activate |
| CNN + Early stopping | 0.94430 | 0.9467 | Early stopping did not activate |

## 5. Detailed Visualization

The report includes several visualizations:

| Figure | File |
| --- | --- |
| MLP learning curve | `codes/figs/mlp_curve.png` |
| CNN learning curve | `codes/figs/cnn_improved_curve.png` |
| CNN momentum learning curve | `codes/figs/opt_momentum_cnn_improved_curve.png` |
| Early stopping CNN curve | `codes/figs/earlystop_cnn_improved_curve.png` |
| Gaussian noise robustness | `codes/figs/robustness_gaussian_noise.png` |
| MLP confusion matrix | `codes/figs/mlp_confusion_matrix.png` |
| CNN confusion matrix | `codes/figs/cnn_confusion_matrix.png` |
| MLP misclassified samples | `codes/figs/mlp_misclassified.png` |
| CNN misclassified samples | `codes/figs/cnn_misclassified.png` |
| MLP first-layer weights | `codes/figs/mlp_first_layer_weights.png` |
| CNN first-layer kernels | `codes/figs/cnn_first_layer_kernels.png` |

## 6. Discussion

The experiments show that the CNN is more suitable than the MLP for MNIST classification. The MLP treats each image as a flat vector, so it does not directly use the two-dimensional spatial structure of digits. In contrast, the CNN applies local convolution filters and shares parameters across spatial locations. This gives it a stronger inductive bias for image classification.

The improved CNN achieved test accuracy `0.9467`, compared with `0.8724` for the MLP baseline. Momentum further improved the CNN to `0.9758`, which was the best result in the project. This indicates that the CNN architecture and optimization method both matter.

Early stopping was less informative in this project. For both MLP and CNN, the validation accuracy did not stop improving within 5 epochs, so the stopping criterion did not activate. Therefore, early stopping did not improve the final result under the current setting.

The robustness experiment showed that CNN is more stable under Gaussian noise. As noise increased, both models degraded, but CNN accuracy dropped more slowly. This supports the conclusion that convolutional features are more robust for image data.

The error analysis visualizations provide qualitative support for the numerical results. The confusion matrices and misclassified examples show where mistakes still occur, while the MLP weight and CNN kernel visualizations show that the two models learn different kinds of representations.

Overall, the final results satisfy the project requirements: the MLP baseline was implemented and evaluated, the CNN operator and model were implemented manually, the MLP and CNN were compared under similar settings, and additional directions were studied with separate results and discussion.
