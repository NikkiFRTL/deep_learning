
def neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred


def w_sum(a, b):
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output


def ele_mul(number, vector):
    output = [0, 0, 0]
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


games = [8.5]
wins = [0.65]
fans = [1.2]
win_condition = [1]
expectation = win_condition[0]
input = [games[0], wins[0], fans[0]]
weights = [0.1, 0.2, -0.1]
alpha = 0.01

# Процесс обучения из 4х итераций.
for iteration in range(4):

    prediction = neural_network(input, weights)
    error = (prediction - expectation) ** 2
    delta = prediction - expectation
    weight_deltas = ele_mul(delta, input)

    print(f'\nIteration: {iteration+1}')
    print(f'Prediction: {float("{:.4f}".format(prediction))}')
    print(f'Error: {float("{:.4f}".format(error))}')
    print(f'Delta: {float("{:.4f}".format(delta))}')
    print(f'Weights: {[float("{:.4f}".format(x)) for x in weights]}')
    print(f'Weight_deltas: {[float("{:.4f}".format(y)) for y in weight_deltas]}')

    # Корректировка весов
    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]



