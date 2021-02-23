
def neural_network(input, weights):
    pred = vector_mat_mul(input, weights)
    return pred


weights = [[0.1, 0.1, -0.3],
           [0.1, 0.2, 0.0],
           [0.0, 1.3, 0.1]]

games = [8.5]
wl_rec = [0.65]
fans = [1.2]
hurt = [0.1]
win = [1]
sad = [0.1]
expectation = [hurt[0], win[0], sad[0]]
input = [games[0], wl_rec[0], fans[0]]
alpha = 0.01
error = [0, 0, 0]
delta = [0, 0, 0]



def w_sum(a, b):
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output


def vector_mat_mul(vector, matrix):
    output = [0, 0, 0]
    for i in range(len(vector)):
        output[i] = w_sum(vector, matrix[i])
    return output


def zeros_matrix(a, b):
     matrix = [[0 for x in range(a)] for x in range(b)]
     return matrix


def outer_prod(vector_a, vector_b):
     out = zeros_matrix(len(vector_a), len(vector_b))
     for i in range(len(vector_a)):
          for j in range(len(vector_b)):
               out[i][j] = vector_a[i] * vector_b[j]
     return out


# Процесс обучения из 4х итераций.
for iteration in range(4):

    prediction = neural_network(input, weights)

    for i in range(len(expectation)):
         error[i] = (prediction[i] - expectation[i]) ** 2
         delta[i] = prediction[i] - expectation[i]

    weight_deltas = outer_prod(input, delta)

    print(f'\nIteration: {iteration+1}')
    print(f'Prediction: {[float("{:.4f}".format(x)) for x in prediction]}')
    print(f'Error: {[float("{:.4f}".format(x)) for x in error]}')
    print(f'Delta: {[float("{:.4f}".format(x)) for x in delta]}')
    print(f'Weights: {[x for x in weights]}')
    print(f'Weight_deltas: {[y for y in weight_deltas]}')

    # Корректировка весов
    for i in range(3):
         for k in range(3):
              weights[i][k] -= alpha * weight_deltas[i][k]
