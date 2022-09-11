# Standard lib
import typing
import logging

# 3rd party
import pandas as pd
import numpy as np

# user
from classifiers import models
from loss import loss_funcs


def training_errors(training_data: pd.DataFrame, classifier: models.BinaryClassifier) -> float:
    errors = 0
    for i, row in training_data.iterrows():
        if classifier.classify(row.features, row.label) < 0:
            errors += 1

    return errors / len(training_data)


def avg_loss(model: models.BinaryClassifier, feature_matrix: np.array, labels: np.array,
             loss_func: typing.Callable = loss_funcs.hinge) -> float:
    """
    @param model:
    @param feature_matrix: numpy array that contains features
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param loss_func: loss function used, default is hinge

    @return: average loss given a model
    """
    loss = 0
    for i in range(0, len(feature_matrix)):
        value = model.classify(feature_matrix[i], labels[i])
        loss += loss_func(value)

    return loss / len(feature_matrix)
