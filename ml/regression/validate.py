# Standard lib
import typing
import logging

# 3rd party
import pandas as pd
import numpy as np

# user
from regression import models
from loss import loss_funcs


def training_errors(regression: models.LinearRegression, training_data: pd.DataFrame) -> float:
    errors = 0
    for i, row in training_data.iterrows():
        if (row.observed_value - regression.predict(row.features)) > 5: # What is a good value here
            errors += 1

    return errors / len(training_data)


def avg_loss(model: models.LinearRegression, training_data: np.array, labels: np.array,
             loss_func: typing.Callable = loss_funcs.squared_error) -> float:
    """
    @param training_data:
    @param regression:  model to test
    @param loss_func: loss function used, default is hinge

    @return: average loss given a model
    """
    loss = 0
    for i in range(0, len(training_data)):
        observed_value = labels[i][0]
        model_value = model.predict(training_data[i][0])
        difference = loss_func(observed_value - model_value)
        loss += difference

    return loss / len(training_data)
