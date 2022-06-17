# Standard lib
import typing
import logging

# 3rd party
import pandas as pd
import numpy as np

# user
from classifiers import models
from loss import loss_funcs


def training_errors(training_data: pd.DataFrame, classifier: models.LinearClassifier) -> float:
    errors = 0
    for i, row in training_data.iterrows():
        if classifier.classify(row.features, row.label) < 0:
            errors += 1

    return errors / len(training_data)


def average_loss(training_data: pd.DataFrame, theta: np.array,
                 loss_func: typing.Callable = loss_funcs.hinge_loss) -> int:
    """
    @param training_data:
    @param theta: a candidate theta as input
    @param loss_func: loss function used, default is hinge

    @return:
    """
    total_error = 0
    for i, row in training_data.iterrows():
        agreement = (theta @ row.features) * row.label
        logging.info(f'agreement value: {agreement}')
        loss_value = loss_func(agreement)
        logging.info(f'loss value: {loss_value}')
        total_error += loss_value

    avg_loss = total_error / len(training_data)
    logging.info(f'total loss: {total_error}')
    logging.info(f'average loss : {avg_loss}')
    return avg_loss
