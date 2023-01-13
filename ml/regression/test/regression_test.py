# 3rd party
import pytest
import numpy as np

# lib
from regression import models, evaluate, train

class TestRegression:

    x_features = np.array([
          [1]
        , [2]
        , [3]
        , [4]
    ])

    labels = np.array([
          [2]
        , [4]
        , [7]
        , [9]
    ])


    def test_avg_loss(self):
        m = models.LinearRegression(theta=np.array([2]))
        actual = evaluate.training_loss(m, self.x_features, self.labels)
        expected = .5
        assert actual == expected
