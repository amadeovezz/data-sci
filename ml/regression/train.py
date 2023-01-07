import random
import logging
from typing import Callable, Tuple

import numpy as np
from numpy import linalg

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
        , mini_batch_number: int = 1
) -> {}:
    """
    An implementation of the gradient descent algorithm for a single variable (sv). This implementation is intended to
    only estimate the coefficient theta (and not the offset)

    @param feature_matrix: numpy array that contains our data (in this case this is a 1x1 matrix)
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param derivative: the derivative of theta for a given loss function, where L(theta;Data).
    Is a function of x (data), y (label), theta (parameter).
    @param init_theta_range: An range of values that theta is randomly initialized from
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
    # Choose random theta to start at
    theta = random.randint(init_theta_range[0], init_theta_range[1])
    theta_progression.append(theta)
    num_of_features = len(feature_matrix)

    # Shuffle training data and labels in unison
    if mini_batch_number >= len(feature_matrix):
        s = f'Mini batch setting uses all features for loss computation.\n Please configure a value that is' \
            f'< number of features.'
        raise Exception(s)
    stochastic_idx = 0
    p = np.random.permutation(len(feature_matrix))
    shuffled_training_data, shuffled_labels = feature_matrix[p], labels[p]

    for epoch in range(1, maximum_num_epochs):
        derivative_sum = 0
        if stochastic:
            for i in range(stochastic_idx, stochastic_idx + mini_batch_number):
                if i >= num_of_features:
                    break
                derivative_sum += derivative(shuffled_training_data[i][0], shuffled_labels[i][0], theta)

            stochastic_idx = stochastic_idx + mini_batch_number
            # Make sure we re-start our index
            stochastic_idx = stochastic_idx % num_of_features
        else:
            # Compute the slope at a specific point on our objective function
            for i in range(0, num_of_features-1):
                derivative_sum += derivative(feature_matrix[i][0], labels[i][0], theta)

        # This is purely for debugging, adds an enormous runtime to the algorithm
        if logging.DEBUG:
            model = models.LinearRegression(np.array([theta]))
            avg_loss = validate.avg_loss(model, feature_matrix, labels)
            logging.debug(f'(theta: {theta}, avg_loss: {avg_loss}) - Slope -> {derivative_sum}')

        # Apply learning rate schedule
        learning_rate = learning_schedule(learning_rate, epoch)

        # Update theta such that loss 'descends'
        theta += learning_rate * (-1 * derivative_sum)

        # Meta data
        theta_progression.append(theta)

        # If slope is small we are close to minimum and stop
        if -.0005 < derivative_sum < .0005:
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

# WIP
def mv_descend(
        feature_matrix: np.array
        , labels: np.array
        , derivative_t: Callable = None
        , derivative_o: Callable = None
        , init_parameter_range: Tuple = (-100, 100)
        , learning_rate: float = .01
        , learning_schedule: Callable = lambda x, y: x
        , maximum_num_epochs: int = 100
        , stochastic: bool = False
        , mini_batch_number: int = 0
) -> {}:
    """
    TODO: extend to > 2 variables.

    An implementation of the gradient descent algorithm multiple variables (mv).

    @param feature_matrix: numpy array that contains our data (in this case this is a 1x1 matrix)
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)
    @param derivative_t: the partial derivative of theta for a given loss function, where L(theta, offset;Data).
    Is a function of x (data), y (label), theta (parameter), theta_0 (offset).
    @param derivative_o: the partial derivative of the offset for a given loss function, where L(theta, offset;Data).
    Is a function of x (data), y (label), theta (parameter), theta_0 (offset).
    @param init_parameter_range: An range of values that the theta, and the offset are randomly initialized from
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
    # Choose random parameters to start at
    r_theta = random.randint(init_parameter_range[0], init_parameter_range[1])
    r_offset = random.randint(init_parameter_range[0], init_parameter_range[1])
    parameters = np.array([r_theta,r_offset])
    theta_progression.append(parameters)

    num_of_features = len(feature_matrix)
    gradient = np.array([0, 0])

    # Shuffle training data and labels in unison
    if mini_batch_number >= len(feature_matrix):
        s = f'Mini batch setting uses all features for loss computation.\n Please configure a value that is' \
            f'< number of features.'
        raise Exception(s)
    stochastic_idx = 0
    p = np.random.permutation(len(feature_matrix))
    shuffled_training_data, shuffled_labels = feature_matrix[p], labels[p]

    for epoch in range(1, maximum_num_epochs):

        np.zeros(gradient)
        sum_partial_theta, sum_partial_offset = 0, 0

        if stochastic:
            for i in range(stochastic_idx, stochastic_idx + mini_batch_number):
                if i >= num_of_features:
                    break
                sum_partial_theta += derivative_t(
                    shuffled_training_data[i][0], shuffled_labels[i][0], parameters[0], parameters[1])
                sum_partial_offset += derivative_o(
                    shuffled_training_data[i][0], shuffled_labels[i][0],parameters[0], parameters[1])

            stochastic_idx = stochastic_idx + mini_batch_number
            # Make sure we re-start our index
            stochastic_idx = stochastic_idx % num_of_features
        else:
            # Compute the slope at a specific point on our objective function
            for i in range(0, num_of_features-1):
                sum_partial_theta += derivative_t(feature_matrix[i][0], labels[i][0], parameters[0], parameters[1])
                sum_partial_offset += derivative_o(feature_matrix[i][0], labels[i][0],parameters[0], parameters[1])

        gradient[0], gradient[1] = sum_partial_theta, sum_partial_offset

        # Apply learning rate schedule
        learning_rate = learning_schedule(learning_rate, epoch)

        # Update parameters elementwise such that loss 'descends'
        np.add(parameters, learning_rate * (-1 * gradient))

        # Meta data
        theta_progression.append(parameters)

        # If norm of gradient is small we are close to minimum and stop
        if -.0005 < linalg.norm(gradient) < .0005:
            total_steps_until_convergence = epoch
            break

    return {
        'model': models.LinearRegression(np.array([round(parameters[0], 5)]), round(parameters[1], 5)),
        'summary': {
            'converged': True if total_steps_until_convergence == maximum_num_epochs else False,
            'total_steps_until_convergence': total_steps_until_convergence,
            'thetas': theta_progression,
        }
    }
