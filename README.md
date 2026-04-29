# CV_HW2: MNIST MLP and CNN from Scratch

This repository contains the code and final report for Project 1 of Neural Network and Deep Learning. The project reconstructs an MLP and a simple CNN on MNIST using NumPy-based implementations.

## Author

- Name: Yuheng Li
- Student ID: 23307130334
- GitHub: yuheng-li-ai

## Repository Structure

```text
codes/
  mynn/
    op.py                 # Linear layer, cross-entropy loss, convolution, activation and pooling operators
    models.py             # MLP and CNN model definitions
    optimizer.py          # SGD and Momentum SGD
    lr_scheduler.py       # Learning-rate schedulers
    runner.py             # Training loop, validation, tqdm progress and early stopping support
  test_train.py           # Main training entry for the selected model
  test_model.py           # Test evaluation entry
  train_opt_momentum.py   # MLP momentum experiment
  train_opt_momentum_cnn.py
  train_earlystop_mlp.py
  train_earlystop_cnn.py
  robustness_gaussian_noise.py
  error_analysis_visualization.py
  weight_visualization.py
final_report.md           # Final report in Markdown format
```

## Notes

The MNIST dataset, checkpoints, logs, and generated figures are not committed to this repository. They are excluded by `.gitignore` because they are generated artifacts or large local files.

Expected local paths after running experiments include:

```text
codes/dataset/
codes/best_models/
codes/figs/
codes/*.log
```

## Main Commands

Run training from the `codes` directory:

```bash
cd codes
python test_train.py
```

Run test evaluation:

```bash
cd codes
python test_model.py
```

Run additional experiments:

```bash
cd codes
python train_opt_momentum.py
python train_opt_momentum_cnn.py
python train_earlystop_mlp.py
python train_earlystop_cnn.py
python robustness_gaussian_noise.py
python error_analysis_visualization.py
```

## Report

The final report is provided in `final_report.md`. It summarizes the MLP baseline, CNN baseline, optimization experiment, early stopping experiment, robustness analysis, and visualization/error analysis.

## Checkpoints

The trained checkpoints used in the report are available on ModelScope:

```text
https://modelscope.cn/models/yuhengli/deeplearning-project1
```
