import numpy as np


def relu(x):
    """
    Возвращает значения > 0 без изменений, а значениям < 0 присваивает 0 умножая х на False.
    """
    return (x > 0) * x


def relu2deriv(output):
    """
    Возвращает 1 (True), если значения > 0 или 0 (False), если значения <= 0.
    """
    return output > 0


# 1. Инициализация весов и данных сети
np.random.seed(1)
streetlights = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
])
# Expectation. Идти или стоять на соответствующую комбинацию светофора, .T - переворот матрицы.
walk_vs_stop = np.array([1, 1, 0, 1]).T
alpha = 0.2
# Количество узлов на layer_1 - комбинаций светофора, в streetlights видно, что их 4 (количество строк).
hidden_size = 4
# Генерация матрицы случайных весов размером 3 х 4
weights_0_1 = 2*np.random.random((3, hidden_size)) - 1
# Генерация матрицы случайных весов размером 4 х 1
weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1

for iteration in range(60):

    # Искомая матрица ошибок на выходе.
    layer_2_error = 0

    for i in range(len(streetlights)):

        # 2. Прогноз + Сравнение
        # [i: i+1] необходимо для создания матрицы [[0, 1, 0]] а не списка [0, 1, 0] в случае [i]
        layer_0 = streetlights[i: i+1]

        # Prediction для layer_1 с обнулением значений < 0
        layer_1 = relu(np.dot(layer_0, weights_0_1))

        # Prediction для layer_2
        layer_2 = np.dot(layer_1, weights_1_2)

        # Подсчет величины ошибки error = (prediction - goal) ** 2 и delta.
        layer_2_error += (layer_2 - walk_vs_stop[i: i+1]) ** 2

        layer_2_delta = layer_2 - walk_vs_stop[i: i+1]

        # 3. Обучение: обратное распространение из layer_2 в layer_1
        # Умножение layer_2_delta и weights_1_2 показывает вклад каждго веса в общую ошибку.
        # relu2deriv фильтрует узлы с незначительным вкладом, пропуская только > 0.
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)

        # 4. Обучение: вычисление приращений и корректировка весов
        weight_delta_1_2 = np.dot(layer_1.T, layer_2_delta)
        weight_delta_0_1 = np.dot(layer_0.T, layer_1_delta)

        weights_1_2 -= alpha * weight_delta_1_2
        weights_0_1 -= alpha * weight_delta_0_1

        print(f"Iteration: {iteration}. Error: {layer_2_error}")
