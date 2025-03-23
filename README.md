### Temporal Models for Predicting in Multi-Horizon at different Prediction Lengths

This repository serves as the summation of my exploratory work on the Jane Street Kaggle Competition. All of the models were trained and optimised to meet the competition's requirements. The competition can be found [here](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/). This concept is highly resonant with Deep Time Index Forecasting model by SalesForce. Although they use a Ridge Regression Model to optimize the target prediction based on online optimization, the training strategy combined with how my model responds to the given context is not similar.


### Introduction
Although I couldn't manage to participate in the competition, I have tried to reimplement the models that I could find in the latest literature. The models are as follows:
- [x] Temporal Fusion Transformer
- [x] Fourier Transformer

My original ideas are as follows:
- [x] Feature Graph Model leveraging Star transformations


### Motivation
Most of the SOTA models available apart from foundational models do not deal with multi-horizon forecasting. Also the lookback window is fixed in most of the models. I wanted to explore the possibility of using a dynamic lookback window and multi-horizon forecasting in the same model.

### Training data and Target

The training data is a time-series data with 78 features(exogenous) and 9 responders(endogenous). The features are anonymised and are not interpretable. The target is responder_6 which ranges from -5 to 5.

### Training Samplers

The data is too large to upload here. For this task writing a DDP custom sampler for pytorch lightning was necessary. However the documentation on this is not very clear. I have written a custom sampler that can be used to train the model on the data. The sampler is available in the `samplers` directory.

### Layers

I have scoured the internet for the latest implementation of most of the layers published in the literature. The layers are available in the `layers` directory.

#### Temporal Fusion Transformer

The Temporal Fusion Transformer is a model that uses a transformer architecture to model the temporal dependencies in the data, although this model is the most robust that I have come across, it was written in tensorflow. I have reimplemented the model in PyTorch Lightning. The model is trained to predict the target at different prediction lengths and Validation score is > 0.3 which is ten times higher than the competition's metric. The model is reorganized to shift from quantile based prediction to predicting the target directly at a specified prediction point.

#### Feature Graph Model
Leveraging the STAR architecture I have tried to implement a model that uses the features as nodes and the target as the central node. The model is trained to predict the target at different prediction lengths and the validation score is > 0.3 which is ten times higher than the competition's metric. This model is not a Graph Neural Network but rather borrows the concept of embedding the features as nodes and the target as the central node.


### Reach

Please reach out to me if you would like to work with me on these ideas. I am open to collaboration and would like to explore these ideas further.