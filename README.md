# SignRecognition
COMP3419 A2 Opt 3 German Traffic Sign Recognition

Algorithm, keep in mind. Definition, in cheatsheet.

## History
Now I have: A simple LEnet-5 model with `train.py` and `predict.py`

Architecture:
- Conv1: 3->16 channels, 3x3 kernel
- Conv2: 16->32 channels, 3x3 kernel
- Conv3: 32->64 channels, 3x3 kernel
- FC1: 1024->128
- FC2: 128->43 classes

<!-- 40% for testing, 60% for training. -->

To do list:
<!-- - Data augmentation (done) -->
<!-- - New dataset? At night (not allowed) -->
- New model? (Shallow CNN/Mini VGG)
  - Ensemble(Adaboost)?
- target common errors?
- Generative-Adversial Network?
- How to benchmark computational cost / improvements in my report?


First, run the model on the entire dataset, and filter out common errors.

Second, run the model on a specific size of images only, and record which size has the most errors.

Third, compare the ablation results.

## Report draft

Spatial Regularization Network works better where one image corresponds to multiple labels, but does not yield significant improvements for one-label. However, in reality, if there are traffic signs that require multiple labels, this might work better than LeNET5.

The dataset was created from approx. 10 hours of video
that was recorded while driving on different road types in
Germany during **daytime**. For real-world applications, night time has to be considered.