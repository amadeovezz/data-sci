# Standard lib
import typing

# 3rd party
import numpy as np

# user
from regression import models


def squared_error(x: float) -> float:
    return x ** 2


def training_loss(model: models.LinearRegression, feature_matrix: np.array, labels: np.array,
                  loss_func: typing.Callable = squared_error) -> float:
    """
    @param model: A linear model
    @param feature_matrix: numpy array that contains features - note this just a 1d array. # TODO: make this a ndarray
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param loss_func: loss function used, default is hinge

    @return: average loss given a model
    """
    loss = 0
    for i in range(0, len(feature_matrix)):
        observed_value = labels[i][0]
        model_value = model.predict(feature_matrix[i])
        difference = loss_func(observed_value - model_value)
        loss += difference

    return loss / len(feature_matrix)
