# 3rd party
import numpy as np


class BinaryClassifier:
    # TODO: maybe make an abstract classifier class
    # TODO: assumes theta is 2 dimensional for learning purposes, make it generalize to n

    theta: np.array = 0
    offset: int = 0

    def __init__(self, theta: np.array, offset: int = 0):
        self.theta = theta
        self.offset = offset

    def __repr__(self)-> str:
        thetas_vars = ['x','y', 'z','t','u','v','w']

        if len(self.theta) > len(thetas_vars):
            return f'__repr__ only works when theta is <= {len(thetas_vars)}'

        abs_thetas = np.abs(self.theta)
        theta_terms = ''
        for i, value in enumerate(abs_thetas):
            if i == 0:
                term = f'{value}{thetas_vars[i]}' if value > 0 else f' - {value}{thetas_vars[i]}'
            else:
                term = f'+ {value}{thetas_vars[i]}' if value > 0 else f' - {value}{thetas_vars[i]}'
            theta_terms += f'{term} '
        return f' {theta_terms}+ {self.offset} = 0'

    def classify(self, feature: np.array, label: int) -> int:
        return np.sign(
            (self.theta.dot(feature) + self.offset) * label
        )

