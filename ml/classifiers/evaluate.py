# Standard lib
import typing
import logging

# 3rd party
import numpy as np

# user
from classifiers import models

def hinge(x: float) -> float:
    if x >= 1:
        return 0
    return 1 - x

def training_loss(model: models.BinaryClassifier, feature_matrix: np.array, labels: np.array,
                  loss_func: typing.Callable = hinge) -> float:
    """
    @param model:
    @param feature_matrix: numpy array that contains features # TODO: make this a ndarray
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param loss_func: loss function used, default is hinge

    @return: average loss given a model
    """
    loss = 0
    for i in range(0, len(feature_matrix)):
        value = model.classify(feature_matrix[i], labels[i])
        loss += loss_func(value)

    return loss / len(feature_matrix)
