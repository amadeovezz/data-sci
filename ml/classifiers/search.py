# standard lib
import logging

# 3rd party
import numpy as np
import pandas as pd

# user
import models


def perceptron(
        training_data: pd.DataFrame
        , t: int
        , theta: np.array = np.array((0, 0), dtype=int)
        , include_offset: bool = True
        , offset: int = 0
) -> {}:
    """
    An implementation of the perceptron algorithm. Assumes theta is two dimensional.

    # TODO: maybe use another data structure besides DataFrames

    @param training_data: a data frame that contains a column named 'features' and a column label
    @param t: max number of iterations of training set, an integer
    @param theta: specify a different initialization of theta, default is 0 vector
    @param include_offset: run the algorithm without an offset, default is True
    @param offset: specify a different initialization of offset, default is 0

    @return: a dictionary with a classifier and some additional meta-data

    Usage:

    results = perceptron(df, 4)
    classifier = results['classifier'] # Get classifier
    summary = results['summary'] # View summary of algorithm

    Or with theta and offset specified:
    linear.perceptron(training_data, 5, theta=np.array((-3,2), dtype=int), offset=-3)
    """

    # Meta-data
    theta_progression = []
    total_errors = 0
    iterations_until_convergence = 0

    misclassified_feature_count = {}
    for _, row in training_data.iterrows():
        misclassified_feature_count[np.array_str(row.features)] = 0

    # Algorithm
    for runs in range(0, t):
        logging.info(f'Iteration through training set: {runs + 1} \n')
        errors_during_training_iteration = 0
        for i, row in training_data.iterrows():
            logging.info(
                f'Iteration through feature {i + 1} - vector: {row.features}, label: {row.label}, theta: {theta}')
            theta_dot_feature = theta.dot(row.features)
            logging.info(
                f'Theta dot feature: {theta_dot_feature},'
                f' label: {row.label},'
                f' product: {theta_dot_feature * row.label}'
            )

            # Decision Boundary logic
            if (theta_dot_feature + offset) * row.label <= 0:
                logging.info(f'Feature is misclassified!')
                theta = theta + (row.label * row.features)
                offset = offset + row.label if include_offset else 0
                logging.info(f'Updated theta and offset...\n')
                # Algorithm logic
                errors_during_training_iteration += 1
                # Meta data
                theta_progression.append(theta)
                total_errors += 1
                misclassified_feature_count[np.array_str(row.features)] += 1
            else:
                logging.info(f'Feature is classified correctly!\n')

        if errors_during_training_iteration == 0:
            iterations_until_convergence = runs + 1
            logging.info(f'No errors found during iteration of training set...\n')
            logging.info(f'Algorithm complete...\n')
            break
        else:
            logging.info(f'Number of errors found during iteration of training set: {errors_during_training_iteration}')
            logging.info(f'-------------------------------------------------\n')

    return {
        'classifier': models.LinearClassifier(theta, offset),
        'summary': {
            'iterations_of_training_data': t,
            'converged': False if iterations_until_convergence == 0 else True,
            'training_iterations_until_convergence': iterations_until_convergence,
            'total_errors': total_errors,
            'thetas': theta_progression,
            'times_features_are_misclassified': misclassified_feature_count
        }
    }


def gradient_descent(
        training_data: pd.DataFrame
        , gradient: np.array = np.array((0, 0), dtype=int)
        , step_size: int = .1
        , maximum_num_steps: int = 1000
        , include_offset: bool = True
        , offset: int = 0
) -> {}:
    """

    WIP!!

    An implementation of the gradient descent algorithm. Assumes theta is two-dimensional.

    @param training_data: a data frame that contains a column named 'features' and a column label
    @param t: max number of iterations of training set, an integer
    @param theta: specify a different initialization of theta, default is 0 vector
    @param include_offset: run the algorithm without an offset, default is True
    @param offset: specify a different initialization of offset, default is 0

    @return: a dictionary with a classifier and some additional meta-data

    """

    # Partial derivatives of average hinge loss wrt theta_1 and theta_2
    # f(theta_1, theta_1) = ( (theta_1 * x_1) + (theta_2 * x_2)) * y)

    partial_derv_theta_1 = lambda x_1, y: -x_1 * y
    partial_derv_theta_2 = lambda x_2, y: -x_2 * y

    # Meta-data
    theta_progression = []
    total_errors = 0

    # Algorithm
    for step_num in range(1, maximum_num_steps):

        # Determine gradient of average hinge loss
        all_gradients = []

        # Get the gradient for each element of training data
        for i, row in training_data.iterrows():
            # calculate partial derivative wrt theta_1
            gradient_theta_1 = partial_derv_theta_1(row.features[0], row.label)
            # calculate partial derivative wrt theta_2
            gradient_theta_2 = partial_derv_theta_2(row.features[1], row.label)
            all_gradients.append([gradient_theta_1, gradient_theta_2])

        # Sum up all gradients
        sum_gradient_theta_1 = 0
        sum_gradient_theta_2 = 0
        for gradient in all_gradients:
            sum_gradient_theta_1 += gradient[0]
            sum_gradient_theta_2 += gradient[1]
        final_gradient = [sum_gradient_theta_1, sum_gradient_theta_2]

        # Get slope of gradient (the norm). Slope tells us how much our function is changing at a given point on the surface

        # If slope is large we are far away from the minimum
        # Take large step size to get closer. Determine how big step size should be

        # If slope is small we are close to the minimum
        # Take small step size

        # Update the gradient
        # Gradient should be negated (moving in the direction of steepest descent)
        # Gradient should be multiplied by the step size
