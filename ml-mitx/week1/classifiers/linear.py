import logging
import typing

import numpy
import numpy as np
import pandas as pd


class Classifier:
    theta = np.array([0, 0])
    theta_0 = 0

    def __init__(self, theta: np.array, theta_0: int=0):
        self.theta = theta
        self.theta_0 = theta_0  # to implement

    def dot(self, feature: np.array) -> int:
        return self.theta @ feature

    def classify(self, feature: np.array, label: int) -> int:
        return np.sign(self.dot(feature) * label)


def perceptron(training_data: pd.DataFrame, t: int) -> {}:
    """
    An implementation of the perceptron algorithm.

    # TODO: add theta_0, maybe use another data structure besides DataFrames

    @param training_data: DataFrame a data frame that contains a column named 'features' and a column label
    @param t: int  max number of iterations of training set, an integer

    @return: a dictionary with a classifier and some additional meta-data

    Usage:
    results = perceptron(df, 4)
    classifier = results['classifier'] # Get classifier
    print(results) # View summary of algorithm
    """

    # Meta-data
    all_thetas = []
    total_errors = 0
    iterations_until_convergence = 0

    # local params required for algorithm
    theta = np.array([0, 0])

    for runs in range(1, t):
        logging.info(f'Iteration through training set: {runs} \n')
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

            if theta_dot_feature * row.label <= 0:
                logging.info(f'Boundary check failed!')
                theta = theta + (row.label * row.features)
                all_thetas.append(theta)
                logging.info(f'Updated theta...\n')
                errors_during_training_iteration += 1
                total_errors += 1
            else:
                logging.info(f'Boundary check passed!\n')

        if errors_during_training_iteration == 0:
            iterations_until_convergence = runs
            logging.info(f'No errors found during iteration of training set...\n')
            logging.info(f'Algorithm complete...\n')
            break
        else:
            logging.info(f'Number of errors found during iteration of training set: {errors_during_training_iteration}')
            logging.info(f'-------------------------------------------------\n')

    return {
        'classifier': Classifier(theta),
        'training_iterations_until_convergence': iterations_until_convergence,
        'total_errors': total_errors,
        'thetas': all_thetas
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
