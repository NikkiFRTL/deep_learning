import numpy as np


ih_weight = np.array([
    [0.3, 0.4, -0.1],
    [0.5, 0.2, 0.1],
    [0.3, 0.3, 0.0]])

# hid
hp_weight = np.array([
    [0.5, 0.2, -0.1],
    [0.4, 0.15, 0.1],
    [0.4, 0.3, 0.0]])

weight = [ih_weight, hp_weight]


def neural_network(input, weights):
    hid = input.dot(weights[0])
    prediction = hid.dot(weights[1])
    return prediction


games = np.array([8.5, 7.7, 4.4])
wins = np.array([0.8, 0.4, 0.1])
fans = np.array([1.2, 3.3, 2.2])
input = np.array([games[0], wins[0], fans[0]])
print(neural_network(input, weight))
