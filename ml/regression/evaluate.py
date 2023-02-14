# Standard lib
import typing

# 3rd party
import numpy as np

# user
from regression import models


def squared_error(x: float) -> float:
    return x ** 2


def training_loss(model: models.LinearRegression, feature_matrix: np.ndarray, labels: np.array,
                  loss_func: typing.Callable = squared_error) -> float:
    """
    @param model: A linear model
    @param feature_matrix: nd array that contains features. Each column represents a feature.
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param loss_func: loss function used, default is hinge

    @return: average loss given a model
    """
    loss = 0
    has_n_features = False if len(feature_matrix.shape) == 1 else True
    num_of_data_points = len(feature_matrix[0, :]) if has_n_features else len(feature_matrix)

    for i in range(0, num_of_data_points):
        observed_value = labels[i]
        data_point = feature_matrix[:, i] if has_n_features else feature_matrix[i]
        model_value = model.predict(data_point)
        difference = loss_func(observed_value - model_value)
        loss += difference

    return loss / num_of_data_points
