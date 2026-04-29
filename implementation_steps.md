# Implementation Steps

## Ground Rules

1. Keep the current baseline results fixed.
2. Do not overwrite `mlp_run` or `cnn_run`.
3. Every new experiment must use its own save directory, log file, and figure name.
4. Record every result separately.
5. Do not mix settings from different directions in one run.

## Existing Fixed Baselines

### MLP Baseline
- Checkpoint: `codes/best_models/mlp_run/best_model.pickle`
- Log: `codes/mlp_train.log`
- Curve: `codes/figs/mlp_curve.png`
- Best validation accuracy: `0.86660`
- Test accuracy: `0.8724`

### CNN Baseline
- Checkpoint: `codes/best_models/cnn_run/best_model.pickle`
- Log: `codes/cnn_train.log`
- Curve: `codes/figs/cnn_curve.png`
- Saved-checkpoint validation accuracy: `0.53630`
- Highest logged dev accuracy: `0.5771`
- Test accuracy: `0.1473`

## Result Separation Rule

For every new experiment, create a separate directory under `codes/best_models`.

Recommended naming:
- `opt_momentum_run`
- `opt_multistep_run`
- `opt_exponential_run`
- `reg_l2_small_run`
- `reg_l2_large_run`
- `reg_dropout_run`
- `aug_translate_run`
- `aug_rotate_run`
- `robust_noise_eval`
- `robust_translate_eval`
- `analysis_outputs`

For each run, also save:
- one log file
- one figure file
- one test result

## Step 1: Prepare Experiment-Specific Train Scripts

Current issue:
- `codes/test_train.py` is only suitable for one active setup at a time.

Recommended action:
1. Keep `codes/test_train.py` as the currently active script only when needed.
2. For cleaner work, create separate experiment scripts later if necessary, for example:
   - `codes/train_opt_momentum.py`
   - `codes/train_reg_l2.py`
   - `codes/train_aug_translate.py`
3. Each script should define its own:
   - model
   - optimizer
   - scheduler if used
   - save directory
   - figure output path

## Step 2: Optimization Direction

### Files to modify
- `codes/mynn/optimizer.py`
- `codes/mynn/lr_scheduler.py`
- one dedicated training script

### What is missing now
- `MomentGD` is empty
- `MultiStepLR` is empty
- `ExponentialLR` is empty

### Implementation order

#### 2.1 Momentum
1. Implement `MomentGD` in `codes/mynn/optimizer.py`.
2. Store velocity for each optimizable layer parameter.
3. Use separate velocity tensors for `W` and `b`.
4. Keep the update rule separate from plain SGD.
5. Train one momentum experiment.
6. Save outputs to:
   - `codes/best_models/opt_momentum_run/`
   - `codes/figs/opt_momentum_curve.png`
   - `codes/opt_momentum.log`
7. Run test evaluation and record the result separately.

#### 2.2 MultiStepLR
1. Implement `MultiStepLR` in `codes/mynn/lr_scheduler.py`.
2. Decide milestone iterations clearly.
3. Keep optimizer as SGD first.
4. Train one multistep experiment.
5. Save outputs to:
   - `codes/best_models/opt_multistep_run/`
   - `codes/figs/opt_multistep_curve.png`
   - `codes/opt_multistep.log`
6. Run test evaluation and record the result separately.

#### 2.3 ExponentialLR
1. Implement `ExponentialLR` in `codes/mynn/lr_scheduler.py`.
2. Use a dedicated run separate from multistep.
3. Save outputs to:
   - `codes/best_models/opt_exponential_run/`
   - `codes/figs/opt_exponential_curve.png`
   - `codes/opt_exponential.log`
4. Run test evaluation and record the result separately.

### What to report separately
- momentum result
- multistep result
- exponential scheduler result
- which one improved or worsened stability

## Step 3: Regularization Direction

### Files to modify
- `codes/mynn/op.py`
- `codes/mynn/models.py`
- one or more dedicated training scripts

### What is currently available
- `Linear` and `conv2D` already support `weight_decay`
- `L2Regularization` class is empty
- no `Dropout` layer exists

### Best implementation order

#### 3.1 L2 Regularization
1. Use the existing `weight_decay` and `weight_decay_lambda` path first.
2. You do not need the empty `L2Regularization` class unless you want an explicit separate layer design.
3. Create at least two L2 runs, for example:
   - small lambda
   - larger lambda
4. Save outputs separately:
   - `codes/best_models/reg_l2_small_run/`
   - `codes/best_models/reg_l2_large_run/`
   - matching figure files and logs
5. Run test evaluation for each and record separately.

#### 3.2 Dropout
1. Add a `Dropout` layer to `codes/mynn/op.py`.
2. The layer must behave differently in training and evaluation.
3. If you add dropout, update model construction in `codes/mynn/models.py`.
4. Add dropout only in a dedicated experiment, not inside the baseline.
5. Save outputs separately:
   - `codes/best_models/reg_dropout_run/`
   - `codes/figs/reg_dropout_curve.png`
   - `codes/reg_dropout.log`
6. Run test evaluation and record separately.

#### 3.3 Early Stopping
1. Only do this if you are willing to change training control logic in `codes/mynn/runner.py`.
2. Add patience and best-checkpoint tracking carefully.
3. Use a dedicated run and do not mix it with dropout or stronger L2.

### What to report separately
- small L2 result
- large L2 result
- dropout result if implemented
- early stopping result if implemented

## Step 4: Data Augmentation Direction

### Files to modify
- likely one new helper script or helper functions inside a new experiment file
- optionally `codes/test_train.py` if you insist on one-script workflow

### What is missing now
- no augmentation pipeline exists
- all data is currently loaded as raw MNIST arrays only

### Best implementation order

#### 4.1 Translation
1. Start with small translation because it is easiest in NumPy.
2. Apply it only to training images.
3. Keep validation and test images clean.
4. Save outputs separately:
   - `codes/best_models/aug_translate_run/`
   - `codes/figs/aug_translate_curve.png`
   - `codes/aug_translate.log`
5. Run test evaluation and record separately.

#### 4.2 Rotation
1. Add small rotation only after translation works.
2. Keep the angle range small.
3. Save outputs separately:
   - `codes/best_models/aug_rotate_run/`
   - `codes/figs/aug_rotate_curve.png`
   - `codes/aug_rotate.log`
4. Run test evaluation and record separately.

#### 4.3 Slight Resizing
1. This is the hardest one in the current NumPy-only structure.
2. Do it last only if earlier augmentations are already complete.

### What to report separately
- translation result
- rotation result
- resizing result if implemented
- comparison with no augmentation baseline

## Step 5: Robustness Analysis

### Files to add or modify
- best done in a separate evaluation script
- reuse saved checkpoints instead of retraining

### Best implementation order

#### 5.1 Gaussian Noise
1. Load one saved model.
2. Evaluate on clean test data.
3. Evaluate on noisy test data.
4. Save the numbers separately for MLP and CNN.

#### 5.2 Translation Perturbation
1. Shift test images slightly.
2. Evaluate MLP and CNN separately.
3. Save the results separately.

#### 5.3 Rotation Perturbation
1. Only do this if you already implemented a stable rotation helper.

### Output format
Use one table per perturbation type:
- clean accuracy
- perturbed accuracy
- accuracy drop

Keep MLP and CNN separated.

## Step 6: Error Analysis and Visualization

### Files to modify or add
- `codes/weight_visualization.py`
- one new script for confusion matrix and misclassified examples is recommended

### Current issues
- `codes/weight_visualization.py` still uses old hard-coded Windows-style paths
- it only partially supports MLP visualization

### Best implementation order

#### 6.1 Confusion Matrix
1. Create confusion matrix generation from test predictions.
2. Save MLP and CNN confusion matrices separately.
3. Use separate output files:
   - `codes/figs/mlp_confusion_matrix.png`
   - `codes/figs/cnn_confusion_matrix.png`

#### 6.2 Misclassified Examples
1. Save several representative mistakes for MLP.
2. Save several representative mistakes for CNN.
3. Do not mix them into one image without labels.

#### 6.3 Weight and Kernel Visualization
1. Fix `codes/weight_visualization.py` paths first.
2. For MLP, visualize first-layer weights reshaped to `28x28`.
3. For CNN, visualize convolution kernels from the first conv layer.
4. Save separately:
   - `codes/figs/mlp_weights.png`
   - `codes/figs/cnn_kernels.png`

## Step 7: Reporting Rule

For every direction, write separate subsections.

Recommended report structure inside Part C and beyond:
1. Optimization
   - Momentum
   - MultiStepLR
   - ExponentialLR
2. Regularization
   - L2 small
   - L2 large
   - Dropout
3. Data augmentation
   - Translation
   - Rotation
4. Robustness analysis
   - Noise
   - Translation perturbation
   - Rotation perturbation
5. Error analysis and visualization
   - Confusion matrix
   - Misclassified examples
   - MLP weights
   - CNN kernels

Do not merge multiple results into one paragraph without labels.

## Recommended Actual Execution Order

1. Implement optimization modules first.
2. Run momentum experiment.
3. Run scheduler experiments.
4. Run L2 regularization experiments.
5. Implement dropout if needed.
6. Implement translation augmentation and test it.
7. Implement robustness evaluation using saved models.
8. Implement confusion matrix and misclassification visualization.
9. Fix and extend weight visualization.
10. Fill the report after each experiment, not all at the end.

## Practical Warning

The current CNN baseline is weak.

So for every later direction:
1. compare MLP-based experiments separately
2. compare CNN-based experiments separately
3. do not assume a change that helps MLP will help CNN
4. do not mix an MLP direction result with a CNN direction result in one sentence without saying which model it belongs to
