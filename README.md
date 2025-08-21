# Emotion recognition model (FER-2013)
The following is the summary of the training and experimentation process across more than 5 notebooks (2 of which are present in this repository)

## Experiment process
### < 40% validation accuracy
The first trials all returned a model that performed badly on the validation set (40% val acc):
- Standard model [3 CNN layers, MaxPool throughout with AvgPool at the end, 2 Dense layers, softmax and cross entropy, adam optimizer] (approximately 100% train acc)
- Added data augmentation (reduced to 50% train acc, due to model not being able to "memorize" images anymore)
- Switching optimizers [RMSProp, Adam, SGD] (relatively little change, Adam performed best)
- Depthwise separable convolutions (no change)
- LR adjustments [0.001 and 0.01] (no change)
- LR schedulers [cosine, cosine with repeats, exponential decay] (no change)

None of which surpassed more than 40% val acc

### 40%-45% validation accuracy
Using the base architecture:
- Batch normalization
- Depthwise separable convolutions
- Global average pooling
- Residual connections

Utilizing Hyperband tuning, we eliminate underperforming models under the following criteria:
- Early stopping (No training loss decrease over 4 epochs)
- Lowest validation accuracy to select models performing well

Searching the right hyperparameters in the following:
- Dropout layers [0.3/0.5 dropout rate, spatial dropout/dense dropout/both]
- Loss functions [crossentropy/focal loss/label smoothing]
- Other LR schedules [ReduceLROnPlateau]

After a search, the model reached the following hyperparameters:


### 50% validation accuracy
Using the parameters above and these additional features:
- More residual connections
- Larger subset of data (80% of all data instead of the previous 20%)
- Longer training (100 epochs)
The model reached

### Transfer learning
Part of the challenge was to develop a model from scratch that could predict the images, but in practice, a pre-trained model could adapt to this dataset (along with numerous others) much better.

### Reflection
Key takeaways:
- Data augmentation is useful
- Transfer learning is very very very useful

Things that I figured would be helpful but I only thought about them after training models:
- Class oversampling in model from scratch (FER-2013 has imbalanced classes so oversampling the minority class would improve performance aside from just using label smoothing)