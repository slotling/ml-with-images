# Emotion recognition model (FER-2013)
## Experiment process
### The dreaded 40% validation accuracy
These trials all returned a model that performed badly on the validation set (40% val acc):
- Standard model [3 CNN layers, MaxPool throughout with AvgPool at the end, 2 Dense layers, softmax and cross entropy, adam optimizer] (approximately 100% train acc)
- Added data augmentation (reduced to 50% train acc)
- Switching optimizers [RMSProp, Adam, SGD] (relatively little change, Adam performed best)
- Depthwise separable convolutions (no change)
- LR adjustments [0.001 and 0.01] (no change)
- LR schedulers [cosine, cosine with repeats, exponential decay] (no change)

### Approaching 60% val acc


### Transfer learning
Part of the challenge was to develop a model from scratch that could predict the images, but in practice, a pre-trained model could adapt to this dataset (along with numerous others) much better.