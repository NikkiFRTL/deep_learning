import numpy as np
import sys
from keras.datasets import mnist


# Функция-активатор.
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


# 1. Загрузка изображений из mnist по шаблоному запросу.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Набор данных для обучения нейронной сети. Картинки, которые нужно разбить на пикселы.
images = x_train[0:1000].reshape(1000, 28*28) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1

labels = one_hot_labels

# Тестовый набор данных для тестирования после обучения.
test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))

for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)
# Корректирующий коэффицент.
alpha = 0.005
# Количество узлов на layer_1.
hidden_size = 100
# Итераций в цикле.
iterations = 300
# Равно количетсву пикселей в изображении 28*28.
pixels_per_image = 784
# Количество возможных меток (0,1,2,3,4,5,6,7,8,9).
num_labels = 10
# Генерация матрицы случайных весов размером 784 х 40
weights_0_1 = 0.2*np.random.random((pixels_per_image, hidden_size)) - 0.1
# Генерация матрицы случайных весов размером 4 х 10
weights_1_2 = 0.2*np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):

    # Создаем переменные ошибки и верных прогнозов.
    error = 0.0
    correct_cnt = 0

    for i in range(len(images)):

        # 2. Прогноз + Сравнение
        # [i: i+1] необходимо для создания матрицы [[0, 1, 0]] а не списка [0, 1, 0] в случае [i]
        # Размер матрицы layer_0 - 1x784
        layer_0 = images[i: i+1]

        # Prediction для layer_1 с обнулением значений < 0
        # Размер матрицы layer_1 - 1x40
        layer_1 = relu(np.dot(layer_0, weights_0_1))

        # Прореживание - dropout. Нужно для уменьшения шума и акцентировании внимания на сигнале.
        # N (в примере 2) - до какого числа от нуля до N, не включительно, будут регенироваться числа.
        # size (в примере 1х40) - возвращает форму, заполненную случайными числами, заданую в size=.
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        # Умножается на 2 для нивелирования потери данных из-за dropout_mask, где случайные данные обращаются 0.
        layer_1 *= dropout_mask * 2

        # Prediction для layer_2
        # Размер матрицы layer_2_delta - 1x10
        layer_2 = np.dot(layer_1, weights_1_2)

        # np.sum нужна для суммирования значений матрицы layer_2
        # Величина ошибки error = (prediction - goal) ** 2 и delta.
        error += np.sum(labels[i: i+1] - layer_2) ** 2
        # Если предсказание маркера верно(True) увеличиваем количество верных предсказаний на 1 (int(True) равно 1)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        # 3. Обучение: обратное распространение из layer_2 в layer_1
        # Умножение layer_2_delta и weights_1_2 показывает вклад каждго веса в общую ошибку.
        # relu2deriv фильтрует узлы с незначительным вкладом, проп ускаятолько > 0.

        # Разности вычмисляются в обратном порядке с нюансами вычисления первого первой.
        # Размер матрицы layer_2_delta - 1x10
        # Размер матрицы weights_1_2.T - 10х40
        # Размер матрицы layer_1_delta  - 1х40
        layer_2_delta = labels[i: i+1] - layer_2
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask

        # 4. Обучение: вычисление приращений и корректировка весов
        weight_delta_1_2 = np.dot(layer_1.T, layer_2_delta)
        weight_delta_0_1 = np.dot(layer_0.T, layer_1_delta)

        weights_1_2 += alpha * weight_delta_1_2
        weights_0_1 += alpha * weight_delta_0_1

    # Определим точность проверки - точность нейронной сети на данных, не участвовавших в ее обучении.
    # Принципы вычислений остаются прежними.
    if j % 10 == 0:
        test_error = 0.0
        test_correct_cnt = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i: i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_error += np.sum(test_labels[i: i + 1] - layer_2) ** 2
            test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))

        sys.stdout.write("\n" +
                         "I: " + str(j) +
                         " Test-Err: " + str(test_error / float(len(test_images)))[0:5] +
                         " Test-Acc: " + str(test_correct_cnt / float(len(test_images))) +
                         " Train-Err: " + str(error / float(len(images)))[0:5] +
                         " Train-Acc: " + str(correct_cnt / float(len(images))))
