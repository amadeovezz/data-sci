import logging
import random
import math

import numpy as np
import pandas as pd

def gradient_descent(
        feature_matrix: np.array
        , labels: np.array
        , step_size: int = .1
        , maximum_num_steps: int = 100
        , include_offset: bool = True
        , offset: int = 0
) -> {}:
    """
    An implementation of the gradient descent algorithm for mean squared error. Assumes theta is two-dimensional.

    @param training_data: a data frame that contains a column named 'features' and a column label
    @param step_size:
    @param maximum_num_steps:
    @param include_offset: run the algorithm without an offset, default is True
    @param offset: specify a different initialization of offset, default is 0

    @return: a trained model


    We have our objective function J which is a surface:

    # J(theta_1, theta_1; training data, labels) = sum 1/2 (observed - (theta_1 * x_1 + theta_2 * x_2))^2

    Looking for the argmin of J. We need to find the gradient

    """

    # Gradients of mean squared error
    partial_derv_theta_1 = lambda t_1, t_2, x_1, x_2, y: (y - (t_1*x_1 + t_1 * x_2)) * x_1
    partial_derv_theta_2 = lambda t_1, t_2, x_1, x_2, y: (y - (t_1*x_1 + t_1 * x_2)) * x_2

    # Meta-data
    theta_progression = []
    total_errors = 0

    # Algorithm
    theta = [0,0]

    for step_num in range(1, maximum_num_steps):

        # Choose random theta to start at
        if step_num == 1:
            theta[0] = random.randint(1, 100)
            theta[1] = random.randint(1, 100)

        # Compute the gradient at a specific point on our surface
        gradient_theta_1 = 0
        gradient_theta_2 = 0

        # We need to sum through our training examples to a specific coordinate:

        for i, feature_vector in enumerate(feature_matrix):
            # calculate partial derivative wrt theta_1
            gradient_theta_1 += partial_derv_theta_1(theta[0], theta[1], feature_vector[0], feature_vector[1], labels[i])
            # calculate partial derivative wrt theta_2
            gradient_theta_1 += partial_derv_theta_2(theta[0], theta[1], feature_vector[0], feature_vector[1], labels[i])

        final_gradient = [gradient_theta_1, gradient_theta_2]
        logging.info(f'Gradient at ({theta[0]}, {theta[1]}): <{gradient_theta_1}, {gradient_theta_2}> ')

        # Update theta, aka move in the direction of the negative gradient
        theta = theta + (step_size * (-1 * final_gradient))
        theta_progression = theta_progression.append(theta)

        # Calculate slope via norm
        slope = math.sqrt(final_gradient[0]**2 + final_gradient[1]**2)

        # Decrease step size
        step_size = 1 / step_num

        # Check the slope of theta via norm
        # If slope is small we are close to minimum and stop
        if slope == 0:
            break

    return theta

