import numpy as np
import sys
from keras.datasets import mnist


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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:1000].reshape(1000, 28*28) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1

test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))

for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)
# Expectation. Идти или стоять на соответствующую комбинацию светофора, .T - переворот матрицы.
alpha = 0.005
# Количество узлов на layer_1 - комбинаций светофора, в streetlights видно, что их 4 (количество строк).
hidden_size = 40

iterations = 350

pixels_per_image = 784

num_labels = 10
# Генерация матрицы случайных весов размером 785 х 40
weights_0_1 = 0.2*np.random.random((pixels_per_image, hidden_size)) - 0.1
# Генерация матрицы случайных весов размером 4 х 10
weights_1_2 = 0.2*np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):

    # Искомая матрица ошибок на выходе.
    error, correct_cnt = (0.0, 0)

    for i in range(len(images)):

        # 2. Прогноз + Сравнение
        # [i: i+1] необходимо для создания матрицы [[0, 1, 0]] а не списка [0, 1, 0] в случае [i]
        layer_0 = images[i: i+1]

        # Prediction для layer_1 с обнулением значений < 0
        layer_1 = relu(np.dot(layer_0, weights_0_1))

        # Prediction для layer_2
        layer_2 = np.dot(layer_1, weights_1_2)

        # Подсчет величины ошибки error = (prediction - goal) ** 2 и delta.
        error += np.sum(labels[i: i+1] - layer_2) ** 2
        #
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        # 3. Обучение: обратное распространение из layer_2 в layer_1
        # Умножение layer_2_delta и weights_1_2 показывает вклад каждго веса в общую ошибку.
        # relu2deriv фильтрует узлы с незначительным вкладом, пропуская только > 0.
        layer_2_delta = labels[i: i+1] - layer_2
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)

        # 4. Обучение: вычисление приращений и корректировка весов
        weight_delta_1_2 = np.dot(layer_1.T, layer_2_delta)
        weight_delta_0_1 = np.dot(layer_0.T, layer_1_delta)

        weights_1_2 += alpha * weight_delta_1_2
        weights_0_1 += alpha * weight_delta_0_1

    sys.stdout.write("\r" + "I:" + str(j) + "Error:" + str(error/float(len(images)))[0:5] +
                     "Correct:" + str(correct_cnt/float(len(images))))
