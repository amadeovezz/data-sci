# 3rd party
import numpy as np


class LinearRegression:
    theta: np.array = 0

    def __init__(self, theta: np.array, offset: float = 0):
        self.theta = theta
        self.offset = offset

    def __repr__(self) -> str:
        theta_values = ['x', 'z', 't', 'u', 'v', 'w']

        if len(self.theta) > len(theta_values):
            return f'__repr__ only works when theta is <= {len(theta_values)}'

        theta_terms = ''
        for i, value in enumerate(self.theta):
            term = f' + {value}{theta_values[i]} ' if value > 0 else f' {value}{theta_values[i]}'
            theta_terms += f'{term} '
        return f'y = {self.offset}{theta_terms} '

    def predict(self, feature_vector: np.array) -> float:
        return self.theta.dot(feature_vector) + self.offset
