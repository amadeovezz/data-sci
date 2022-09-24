import random
import logging
from typing import Callable

import numpy as np

from regression import models, validate


def sv_descend(
        feature_matrix: np.array
        , labels: np.array
        , derivative: Callable = None
        , step_size: float = .1
        , maximum_num_steps: int = 100
) -> {}:
    """

    An implementation of the gradient descent algorithm for a single variable (sv). This implementation only estimates
    one parameter .

    @param feature_matrix: numpy array that contains our data (in this case this is a 1x1 matrix)
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param callable: the derivative of theta
    @param step_size: how much to nudge theta
    @param maximum_num_steps: max number of steps

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

    for step_num in range(1, maximum_num_steps):

        # Choose random theta to start at
        if step_num == 1:
            theta = random.randint(-10, 10)

        # Compute the slope at a specific point on our curve
        slope = 0

        # We need to sum through our training examples at specific coordinate:
        for i in range(0, len(feature_matrix)):
            slope += derivative(feature_matrix[i][0], labels[i][0], theta)

        # This is purely for debugging, adds an enormous runtime to the algorithm
        if logging.INFO:
            model = models.LinearRegression(np.array([theta]))
            avg_loss = validate.avg_loss(model, feature_matrix, labels)
            logging.info(f'(theta: {theta}, avg_loss: {avg_loss}) - Slope -> {slope}')

        # Update theta, aka move in the direction of the negative slope
        theta += step_size * (-1 * slope)

        # Meta data
        theta_progression.append(theta)

        # Decrease step size
        step_size = step_size * 3/4

        # Check the slope of theta via norm
        # If slope is small we are close to minimum and stop
        if -.5 < slope < .5:
            total_steps_until_convergence = step_num
            break

    return {
        'model': models.LinearRegression(np.array([round(theta,5)])),
        'summary': {
            'converged': True if total_steps_until_convergence == maximum_num_steps else False,
            'total_steps_until_convergence': total_steps_until_convergence,
            'thetas': theta_progression,
        }
    }