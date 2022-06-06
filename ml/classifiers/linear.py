import logging
import typing

import numpy
import numpy as np
import pandas as pd


class Classifier:
    theta = np.array([0, 0])
    offset = 0

    def __init__(self, theta: np.array, offset: int = 0):
        self.theta = theta
        self.offset = offset

    def dot(self, feature: np.array) -> int:
        return self.theta @ feature

    def classify(self, feature: np.array, label: int) -> int:
        return np.sign(
            (self.dot(feature) + self.offset) * label
        )


def perceptron(training_data: pd.DataFrame, t: int, theta: np.array = np.array((0, 0), dtype=int), offset: int = 0) -> {}:
    """
    An implementation of the perceptron algorithm. Assumes theta is two dimensional.

    # TODO: maybe use another data structure besides DataFrames

    @param training_data: a data frame that contains a column named 'features' and a column label
    @param t: max number of iterations of training set, an integer
    @param theta: specify a different initialization of theta, default is 0 vector
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
            theta_dot_feature = theta @ row.features
            logging.info(
                f'Theta dot feature: {theta_dot_feature},'
                f' label: {row.label},'
                f' product: {theta_dot_feature * row.label}'
            )

            if (theta_dot_feature + offset) * row.label <= 0:
                # Boundary logic
                logging.info(f'Feature is misclassified!')
                theta = theta + (row.label * row.features)
                offset = offset + row.label
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
        'classifier': Classifier(theta, offset),
        'summary': {
            'iterations_of_training_data': t,
            'converged': False if iterations_until_convergence == 0 else True,
            'training_iterations_until_convergence': iterations_until_convergence,
            'total_errors': total_errors,
            'thetas': theta_progression,
            'times_features_are_misclassified': misclassified_feature_count
        }
    }


def training_errors(training_data: pd.DataFrame, classifier: Classifier) -> int:
    errors = 0
    for i, row in training_data.iterrows():
        if classifier.classify(row.features, row.label) < 0:
            errors += 1

    return errors / len(training_data)


def hinge_loss(x: int) -> int:
    if x >= 1:  # Dot product agrees with label
        return 0
    return 1 - x  # Dot product does not agree with label


def objective_function(training_data: pd.DataFrame, theta: numpy.array, loss_func: typing.Callable = hinge_loss) -> int:
    """
    @param training_data: parameter to objective function
    @param theta: input
    @param loss_func: loss function used, default is hinge

    @return:
    """
    sum = 0
    for i, row in training_data.iterrows():
        agreement = (theta @ row.features) * row.label
        logging.info(f'agreement value: {agreement}')
        loss_value = loss_func(agreement)
        logging.info(f'loss value: {loss_value}')
        sum += loss_value

    output = sum / len(training_data)
    logging.info(f'total_sum: {sum}')
    logging.info(f'output of value of objective function: {output}')
    return output
