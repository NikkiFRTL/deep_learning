import numpy as np


def neural_network(input, weights):
    pred = input.dot(weights)
    return pred


games = np.array([8.5, 9.0, 8.5])
wins = np.array([0.65, 0.3, 0.77])
fans = np.array([1.2, 1.0, 0.6])
weights = np.array([0.1, 0.2, 0])
input = np.array([games[0], wins[0], fans[0]])

prediction = neural_network(input, weights)
print(prediction)
