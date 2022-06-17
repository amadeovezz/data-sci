# Standard lib
import typing
import logging

# 3rd party
import pandas as pd

# user
from regression import models
from loss import loss_funcs


def training_errors(training_data: pd.DataFrame, regression: models.LinearRegression) -> float:
    errors = 0
    for i, row in training_data.iterrows():
        if (row.observed_value - regression.predict(row.features)) > 5: # What is a good value here
            errors += 1

    return errors / len(training_data)


def average_loss(training_data: pd.DataFrame, regression: models.LinearRegression,
                 loss_func: typing.Callable = loss_funcs.hinge_loss) -> int:
    """
    @param training_data:
    @param regression:  model to test
    @param loss_func: loss function used, default is hinge

    @return:
    """
    total_error = 0
    for i, row in training_data.iterrows():
        difference = (row.observed_value - regression.predict(row.features))
        logging.info(f'difference: {difference}')
        loss_value = loss_func(difference)
        logging.info(f'loss value: {loss_value}')
        total_error += loss_value

    avg_loss = total_error / len(training_data)
    logging.info(f'total loss: {total_error}')
    logging.info(f'average loss : {avg_loss}')
    return avg_loss
