# Emotion recognition model (FER-2013)
The following is the summary of the training and experimentation process across more than 5 notebooks (2 of which are present in this repository)

## Basic iterative process
1. Check if pipeline works
2. Quickly train a bunch of models to pick the most performant one (Modify architecture, then use Hyperband tuning for hyperparameter search)
3. Final training and evaluation

## Experiment process
### <= 40% validation accuracy
The first trials all returned a model that performed badly on the validation set:
- Standard model [3 CNN layers, MaxPool throughout with AvgPool at the end, 2 Dense layers, softmax and cross entropy, adam optimizer] (approximately 100% train acc, around 40% val acc)
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

After a search, the tuner reached the following hyperparameters:
- Spatial dropout: `0.04`
- Dense dropout: `0.58`
- Loss function: `tf.keras.losses.CategoricalCrossentropy()`
- Label smoothing: `0.34`

The model reached 42% validation accuracy - no improvement.

### 50% validation accuracy
Using the parameters above and these additional features:
- More residual connections
- Larger subset of data (80% of all data instead of the previous 20%)
- Longer training (100 epochs)
The model finally reached 52% accuracy

## Transfer learning
Part of the challenge was to develop a model from scratch that could predict the images, but in practice, a pre-trained model could adapt to this dataset (along with numerous others) much better.

### Standard model
The ResNet50 backbone is chosen at a baseline. The head is global average pooling and some dense layers.

After 50 epochs, the accuracy on a subset of data is 24.3%.

### subheading here


### Reflection
Key takeaways:
- Data augmentation is useful
- Transfer learning is very very very useful

Things that I figured would be helpful but I only thought about them after training models:
- Class oversampling in model from scratch (FER-2013 has imbalanced classes so oversampling the minority class would improve performance aside from just using label smoothing)
- Use `model.evaluate()` to evaluate models, don't use the stats from the last epoch.