import random
import logging
from typing import Callable, Tuple

import numpy as np

from regression import models, validate


def sv_descend(
        feature_matrix: np.array
        , labels: np.array
        , derivative: Callable = None
        , init_theta_range: Tuple = (-100, 100)
        , learning_rate: float = .01
        , learning_schedule: Callable = lambda x, y: x
        , maximum_num_epochs: int = 100
        , stochastic: bool = False
        , mini_batch_number: int = 0
) -> {}:
    """
    An implementation of the gradient descent algorithm for a single variable (sv). This implementation is intended to
    only estimate the coefficient theta (and not the offset)

    @param feature_matrix: numpy array that contains our data (in this case this is a 1x1 matrix)
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param derivative: the derivative of theta, must be a function of (x,y,theta), where x is a data point, y is an
    observed value and theta is the parameter
    @param init_theta_range: An range of values that the theta is randomly initialized from
    @param learning_rate: proportionality constant for update to theta
    @param learning_schedule: apply a custom function to the learning rate, must be a function of (learning_rate and epoch)
    @param maximum_num_epochs: max number of epochs
    @param stochastic: enable stochastic gradient descent
    @param mini_batch_number: the number of random features to use for stochastic gradient descent

    @return: a dictionary with a model estimated and some additional meta-data

    Usage:

    results = gradient_descent(training_data, labels, derivative)
    model = results['model'] # Get model
    summary = results['summary'] # View summary of algorithm

    """

    # Meta-data
    theta_progression = []
    total_steps_until_convergence = 0

    # Algorithm
    theta = 0
    num_of_features = len(feature_matrix)-1

    # Shuffle training data and labels in unison
    if stochastic:
        stochastic_idx = 0
        p = np.random.permutation(len(feature_matrix))
        shuffled_training_data, shuffled_labels = feature_matrix[p], labels[p]


    for epoch in range(1, maximum_num_epochs):
        # Choose random theta to start at
        if epoch == 1:
            theta = random.randint(init_theta_range[0], init_theta_range[1])
            theta_progression.append(theta)

        slope = 0
        if stochastic:
            for i in range(stochastic_idx, stochastic_idx + mini_batch_number):
                slope += derivative(shuffled_training_data[i][0], shuffled_labels[i][0], theta)
            if mini_batch_number == 0:
                stochastic_idx += 1
            else:
                stochastic_idx = stochastic_idx + mini_batch_number
        else:
            # Compute the slope at a specific point on our objective function
            for i in range(0, num_of_features):
                slope += derivative(feature_matrix[i][0], labels[i][0], theta)

        # This is purely for debugging, adds an enormous runtime to the algorithm
        if logging.DEBUG:
            model = models.LinearRegression(np.array([theta]))
            avg_loss = validate.avg_loss(model, feature_matrix, labels)
            logging.debug(f'(theta: {theta}, avg_loss: {avg_loss}) - Slope -> {slope}')

        # Update theta such that loss 'descends'
        theta += learning_rate * (-1 * slope)

        # Meta data
        theta_progression.append(theta)

        # Apply learning rate schedule
        learning_rate = learning_schedule(learning_rate, epoch)

        # If slope is small we are close to minimum and stop
        if -.0005 < slope < .0005:
            total_steps_until_convergence = epoch
            break

    return {
        'model': models.LinearRegression(np.array([round(theta, 5)])),
        'summary': {
            'converged': True if total_steps_until_convergence == maximum_num_epochs else False,
            'total_steps_until_convergence': total_steps_until_convergence,
            'thetas': theta_progression,
        }
    }
