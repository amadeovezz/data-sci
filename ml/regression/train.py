import random
import logging
from typing import Callable, Tuple

import numpy as np
from numpy import linalg

from regression import models, evaluate

def sp_descend(
        feature_matrix: np.array
        , labels: np.array
        , derivative_per_data_point: Callable = None
        , init_theta_range: Tuple = (-100, 100)
        , learning_rate: float = .01
        , learning_schedule: Callable = lambda x, y: x
        , maximum_num_epochs: int = 100
        , stochastic: bool = False
        , mini_batch_number: int = 1
) -> {}:
    """
    An implementation of gradient descent for a single parameter (sp).
    This implementation is intended to only estimate the coefficient theta in a linear model: y = theta * x.

    Only supports 1D features.

    @param feature_matrix: numpy array that contains our data (in this case this is a 1xn matrix)
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)

    @param derivative_per_data_point: the derivative of theta with respect to a given loss function.  L(theta;Data).
    Only intended for average squared error. This value must be summed up and multiplied by 1/num_of_features.

    @param init_theta_range: An range of values that theta is randomly initialized from
    @param learning_rate: proportionality constant for update to theta
    @param learning_schedule: apply a custom function to the learning rate, must be a function of
    learning_rate and epoch. Ie: lambda learning_rate, epoch: return learning_rate * epoch
    @param maximum_num_epochs: max number of epochs
    @param stochastic: enable stochastic partial_derivatives descent
    @param mini_batch_number: the number of random features to use for stochastic partial_derivatives descent

    @return: a dictionary with a model estimated and some additional meta-data

    Usage:

    results = sp_descend(training_data, labels, derivative)
    model = results['model'] # Get model
    summary = results['summary'] # View summary of algorithm

    """

    # Meta-data
    theta_progression = []
    total_steps_until_convergence = 0
    num_of_features = len(feature_matrix)

    # Algorithm
    # Choose random theta to start at
    theta = random.randint(init_theta_range[0], init_theta_range[1])
    theta_progression.append(theta)

    # Shuffle training data and labels in unison
    if mini_batch_number >= len(feature_matrix):
        s = f'Mini batch setting uses all features for loss computation.\n Please configure a value that is' \
            f'< number of features.'
        raise Exception(s)
    stochastic_idx = 0
    p = np.random.permutation(len(feature_matrix))
    shuffled_training_data, shuffled_labels = feature_matrix[p], labels[p]

    for epoch in range(1, maximum_num_epochs):
        derivative = 0
        if stochastic:
            for i in range(stochastic_idx, stochastic_idx + mini_batch_number):
                if i >= num_of_features:
                    break
                derivative += derivative_per_data_point(shuffled_training_data[i], shuffled_labels[i], theta)

            stochastic_idx = stochastic_idx + mini_batch_number
            # Make sure we re-start our index
            stochastic_idx = stochastic_idx % num_of_features
        else:
            # Compute the slope at a specific point on our objective function
            for i in range(0, num_of_features-1):
                derivative += derivative_per_data_point(feature_matrix[i], labels[i], theta)

        # average least squares (TODO: this is specific for least squares - eventually generalize)
        derivative = 1/num_of_features * derivative

        # This is purely for debugging, adds an enormous runtime to the algorithm
        if logging.DEBUG:
            model = models.LinearRegression(np.array([theta]))
            avg_loss = evaluate.training_loss(model, feature_matrix, labels)
            logging.debug(f'(theta: {theta}, avg_loss: {avg_loss}) - Slope -> {derivative}')

        # Apply learning rate schedule
        learning_rate = learning_schedule(learning_rate, epoch)

        # Update theta such that loss 'descends'
        theta += learning_rate * (-1 * derivative)

        # Meta data
        theta_progression.append(theta)

        # If slope is small we are close to minimum and stop
        if -.0005 < derivative < .0005:
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

def tp_descend(
        feature_matrix: np.array
        , labels: np.array
        , derivative_t_per_data_point: Callable = None
        , derivative_o_per_data_point: Callable = None
        , init_parameter_range: Tuple = (-100, 100)
        , learning_rate: float = .0001
        , learning_schedule: Callable = lambda x, y: x
        , maximum_num_epochs: int = 100
        , stochastic: bool = False
        , mini_batch_number: int = 0
) -> {}:
    """

    An implementation of gradient descent for two parameters (tp).
    This implementation is intended to only estimate the coefficient theta in a linear model: y = theta * x + theta_0

    @param feature_matrix: numpy array that contains our data (only supports 1d features)
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)

    @param derivative_t_per_data_point: the partial derivative of theta with respect to a given loss function,
    where L(theta, offset;Data). Only intended for average squared error.

    @param derivative_o_per_data_point: the partial derivative of the offset with respect to a given loss function,
    where L(theta, offset;Data). Only intended for average squared error.

    @param init_parameter_range: An range of values that the theta, and the offset are randomly initialized from
    @param learning_rate: proportionality constant for update to theta
    @param learning_schedule: apply a custom function to the learning rate, must be a function of
    learning_rate and epoch. Ie: lambda learning_rate, epoch: return learning_rate * epoch
    @param maximum_num_epochs: max number of epochs
    @param stochastic: enable stochastic gradient descent
    @param mini_batch_number: the number of random features to use for stochastic gradient descent

    @return: a dictionary with a model estimated and some additional meta-data

    Usage:
    results = tp_descend(training_data, labels, derivative)
    model = results['model'] # Get model
    summary = results['summary'] # View summary of algorithm
    """

    # Meta-data
    parameter_progression = []
    total_steps_until_convergence = 0

    # Algorithm
    # Choose random parameters to start at
    r_theta = random.randint(init_parameter_range[0], init_parameter_range[1])
    r_offset = random.randint(init_parameter_range[0], init_parameter_range[1])
    parameters = np.array([r_theta,r_offset])
    parameter_progression.append(parameters)

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

        gradient = np.array([0, 0])

        if stochastic:
            for i in range(stochastic_idx, stochastic_idx + mini_batch_number):
                if i >= num_of_features:
                    break
                gradient[0] += derivative_t_per_data_point(
                    shuffled_training_data[i], shuffled_labels[i], parameters[0], parameters[1])
                gradient[1] += derivative_o_per_data_point(
                    shuffled_training_data[i], shuffled_labels[i],parameters[0], parameters[1])

            stochastic_idx = stochastic_idx + mini_batch_number
            # Make sure we re-start our index
            stochastic_idx = stochastic_idx % num_of_features
        else:
            # Compute the slope at a specific point on our objective function
            for i in range(0, num_of_features-1):
                gradient[0] += derivative_t_per_data_point(feature_matrix[i], labels[i], parameters[0], parameters[1])
                gradient[1] += derivative_o_per_data_point(feature_matrix[i], labels[i], parameters[0], parameters[1])

        # avg_loss
        gradient = 1/num_of_features * gradient

        # This is purely for debugging, adds an enormous runtime to the algorithm
        if logging.DEBUG:
            model = models.LinearRegression(np.array([round(parameters[0], 5)]), round(parameters[1], 5))
            avg_loss = evaluate.training_loss(model, feature_matrix, labels)
            logging.debug(f'(theta: {parameters[0]}, theta_0:{parameters[1]}  avg_loss: {avg_loss}) - Gradient -> {gradient}')


        # Apply learning rate schedule
        learning_rate = learning_schedule(learning_rate, epoch)

        # Update parameters elementwise such that loss 'descends'
        parameters = np.add(parameters, learning_rate * (-1 * gradient))

        # Meta data
        parameter_progression.append(parameters)

        # If norm of gradient is small we are close to minimum and stop
        if -.0005 < linalg.norm(gradient) < .0005:
            total_steps_until_convergence = epoch
            break

    return {
        'model': models.LinearRegression(np.array([round(parameters[0], 5)]), round(parameters[1], 5)),
        'summary': {
            'converged': True if total_steps_until_convergence == maximum_num_epochs else False,
            'total_steps_until_convergence': total_steps_until_convergence,
            'parameters': parameter_progression,
        }
    }


def mp_descend(
        feature_matrix: np.ndarray
        , labels: np.array
        , init_parameter_range: Tuple = (-100, 100)
        , learning_rate: float = .0001
        , learning_schedule: Callable = lambda x, y: x
        , maximum_num_epochs: int = 100
        , stochastic: bool = False
        , mini_batch_number: int = 0
) -> {}:
    """

    An implementation of gradient descent for a linear model with multiple parameters and features.
    A linear model in the form of: y = w_0 x_0 * w_1  x_2 * ... * w_n x_n + b.
    TODO: add stochastic parameter

    Only supports optimization for average squared error.

    @param feature_matrix: numpy 2d array that contains our data, each column is considered one feature
    @param labels: numpy array containing the labels associated with the feature matrix (assume y^i is associated with x^i)

    @param init_parameter_range: An range of values that the theta, and the offset are randomly initialized from
    @param learning_rate: proportionality constant for update to theta
    @param learning_schedule: apply a custom function to the learning rate, must be a function of
    learning_rate and epoch. Ie: lambda learning_rate, epoch: return learning_rate * epoch
    @param maximum_num_epochs: max number of epochs
    @param stochastic: enable stochastic partial_derivatives descent
    @param mini_batch_number: the number of random features to use for stochastic partial_derivatives descent

    @return: a dictionary with a model estimated and some additional meta-data

    Usage:

    results = mp_descend(training_data, labels, derivative)
    model = results['model'] # Get model
    summary = results['summary'] # View summary of algorithm

    """

    # Meta-data
    parameter_progression = []
    total_steps_until_convergence = 0
    num_of_features = len(feature_matrix[:,0])

    # Algorithm
    # Choose random parameters to start at
    weights = np.random.uniform(low=init_parameter_range[0], high=init_parameter_range[1], size=(num_of_features,))
    bias = random.randint(init_parameter_range[0], init_parameter_range[1])
    parameter_progression.append({'weights': weights, 'bias': bias})

    for epoch in range(1, maximum_num_epochs):

        gradients_of_weights = np.zeros(num_of_features)
        gradient_of_bias = 0

        for i in range(0, num_of_features - 1):
            feature, label = feature_matrix[:,i], labels[i]
            u = np.dot(weights, feature) + bias
            v = np.full( (1, num_of_features), u)
            gradients_of_weights += 2 * feature * -1 * label * v
            gradient_of_bias += 2 * -1 * label * u

        # avg_loss
        gradients_of_weights = 1/num_of_features * gradients_of_weights
        gradient_of_bias = 1/num_of_features * gradient_of_bias

        # Apply learning rate schedule
        learning_rate = learning_schedule(learning_rate, epoch)

        # Update parameters elementwise such that loss 'descends'
        weights = np.add(weights, learning_rate * (-1 * gradients_of_weights))
        bias = bias + (learning_rate * (-1 * gradient_of_bias))

        # Meta data
        parameter_progression.append({'weights': weights, 'bias': bias})

        # If norm of gradient is small we are close to minimum and stop
        all_gradients = np.zeros(num_of_features + 1)
        all_gradients[0:num_of_features] = weights
        all_gradients[-1] = bias
        if -.0005 < np.linalg.norm(all_gradients) < .0005:
            total_steps_until_convergence = epoch
            break


    return {
        'model': models.LinearRegression(np.round(weights, decimals=5), round(bias, 5)),
        'summary': {
            'converged': True if total_steps_until_convergence == maximum_num_epochs else False,
            'total_steps_until_convergence': total_steps_until_convergence,
            'parameters': parameter_progression,
        }
    }
