# Submission Checklist

## A. Submission Materials
- [ ] Submit one single PDF report to eLearning.
- [ ] Put my name in the report.
- [ ] Put my student ID in the report.
- [ ] Include a Github link to the code in the report.
- [ ] Include a link to trained model weights or saved checkpoints in the report.
- [ ] Make sure the Github repository does not contain the dataset.
- [ ] Make sure the Github repository does not contain trained weights.
- [ ] Make sure the Github repository does not contain other large files.
- [ ] Confirm the report matches the actual implementation.
- [ ] Confirm the code can be run.
- [ ] Submit before 11:59 PM, May 24, 2026.

## B. Global Project Constraints
- [ ] Use only the provided MNIST dataset.
- [ ] Do not use any external dataset.
- [ ] Implement my own required operators and models.
- [ ] Do not use deep learning modules that directly solve the required tasks.
- [ ] Keep experiments focused on clear and fair comparison rather than many unrelated tricks.

## C. Part A: MLP Baseline
- [ ] Implement `Linear.forward`.
- [ ] Implement `Linear.backward`.
- [ ] Implement cross-entropy loss with softmax.
- [ ] Train an MLP on MNIST.
- [ ] Report training performance.
- [ ] Report validation performance.
- [ ] Include at least one learning curve in the report.

## D. Part B: CNN Model
- [ ] Implement `conv2D` by myself.
- [ ] Build a simple CNN for MNIST classification.
- [ ] Compare CNN with the MLP baseline under reasonable and fair settings.
- [ ] Discuss why CNN performs better or worse than MLP.

## E. Part C: Two Additional Directions
- [ ] Choose exactly two directions after finishing Part A and Part B.
- [ ] For direction 1, explain what was changed or analyzed.
- [ ] For direction 1, compare with an appropriate baseline.
- [ ] For direction 1, discuss the conclusion clearly.
- [ ] For direction 2, explain what was changed or analyzed.
- [ ] For direction 2, compare with an appropriate baseline.
- [ ] For direction 2, discuss the conclusion clearly.

Optional directions listed in the PDF:
- [ ] Optimization
- [ ] Regularization
- [ ] Data augmentation
- [ ] Robustness analysis
- [ ] Error analysis and visualization

## F. Experimental Principles
- [ ] Change one major factor at a time.
- [ ] Keep MLP and CNN training settings as similar as possible when comparing them.
- [ ] Do not change many hyperparameters simultaneously when studying one modification.
- [ ] Focus on clear and convincing conclusions rather than maximum accuracy.

## G. Report Contents Required by the PDF
- [ ] Section: MLP baseline
- [ ] Section: CNN model and MLP-vs-CNN comparison
- [ ] Section: Two additional directions
- [ ] Section: Main results table
- [ ] Section: Detailed visualization
- [ ] Section: Discussion

## H. Visualization Candidates Mentioned in the PDF
- [ ] Learning curve
- [ ] Confusion matrix
- [ ] Weight visualization
- [ ] Convolution kernel visualization
