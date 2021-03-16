import numpy as np
import sys
from keras.datasets import mnist


def get_image_section(layer, row_from, row_to, col_from, col_to):
    """
    Начнем с прямого распространения.
    Метод демонстрирует, как выбрать подобласть в пакете изображений.
    """
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to-row_from, col_to-col_from)


# Функции-активаторы:
# Для hidden_layer.
def tanh(x):
    return np.tanh(x)


# Для обратного распространения.
def tanh2deriv(output):
    return 1 - (output ** 2)


# Для выходного слоя.
def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


np.random.seed(1)
# 1. Загрузка изображений из mnist по шаблоному запросу.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Набор данных для обучения нейронной сети. Картинки, которые нужно разбить на пикселы.
images = x_train[0:1000].reshape(1000, 28*28) / 255
labels = y_train[0:1000]
one_hot_label = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_label[i][l] = 1
labels = one_hot_label
# Тестовый набор данных для тестирования после обучения.
test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

# Корректирующий коэффицент.
alpha = 2
# Итераций в цикле.
iterations = 300
# Равно количетсву пикселей в изображении (28*28).
pixels_per_image = 784
# Количество возможных меток (0,1,2,3,4,5,6,7,8,9)
num_labels = 10
# Размер для разбиения изображений на группы(кучки) для обработки за 1 итерацию для ускорения работы
# нейронной сети, обучение и точность прогноза растет плавно.
batch_size = 128
# Количество рядов(пикселей) в одном входящем изображении.
input_rows = 28
# Количество столбцов(пикселей) в одном входящем изображении.
input_cols = 28
# Размеры одного ядра (строк, столбцов).
kernel_rows = 3
kernel_cols = 3
# Количетсво ядер.
num_kernels = 16
# Количество узлов на layer_1 = 10000
hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels
# Генерация матрицы случайных весов для ядер(от -0.01 до 0.01) размером (9, 16).
kernels = 0.02*np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
# Генерация матрицы случайных весов (от -0.1 до 0.1) размером (10000, 10).
weights_1_2 = 0.2*np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    correct_cnt = 0
    for i in range(int(len(images) / batch_size)):
        batch_start = i * batch_size
        batch_end = (i+1) * batch_size
        # Матрица размером (128, 784)
        layer_0 = images[batch_start:batch_end]
        # Подготовка входных данных для сверточного слоя.
        # Здесь layer_0 - это пакет 128 изображений 28 x 28.
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

        # Сверточный слой (Convolutional layer).
        # Вместо одного большого, плотного, линейного слоя, связывающего каждый вход с каждым выходом,
        # использ. в каждой позиции на входе множество очень маленьких линейных слоев, где не более 25 входов и 1 выход.
        # Улучшает устройчивость модели к переобучению на обучающих данных и улучшает ее способность к обобщению.
        # Этот прием позволяет каждому ядру изучить определенную закономерность и затем обнаруживать ее
        # везде, где она присутствует в изображении.
        sects = list()
        # Так как подобласть выбирается в пакете входных изображений, мы должны вызвать get_image_section
        # для каждой позиции в изображении.
        # Цикл for последовательно выбирает подобласти (kernel_rows х kernel_cols) в изображениях и помещает их
        # в список sects.
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0,
                                         row_start,
                                         row_start+kernel_rows,
                                         col_start,
                                         col_start+kernel_cols)
                sects.append(sect)

        # Далее этот список объединяется построчно.
        expanded_input = np.concatenate(sects, axis=1)
        # Форма es -  (128, 625, 3, 3). Которую нужно преобразовать в (128*625) х (3*3) = 7200000.
        es = expanded_input.shape
        # Форма flattened_input - (80000, 9).
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        # Каждая подобласть — это отдельное изображение.
        # Таким образом, условно для 8 изображений в пакете и 100 подобластей в каждом мы получим
        # 800 изображений меньшего размера. Передача их в прямом направлении
        # через линейный слой с одним выходным нейроном равносильна получению
        # прогноза из этого линейного слоя для каждой подобласти в каждом пакете.
        # Если, напротив, передать изображения в линейный слой с N выходными нейронами,
        # на выходе мы получим тот же результат, что и при использовании
        # N линейных слоев (ядер) в каждой позиции в изображении.
        # Форма kernel_output - (80000, 16).
        kernel_output = np.dot(flattened_input, kernels)

        # Прогноз для layer_1 с применением функции-активатора tanh.
        # Размер матрицы layer_1 - (128, 1000).
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        # Реализация фильтрации (обнуления случайных значений умножением на матрицу со случайными значениями 0 или 1)
        # для снижения переобучения нейронной сети.
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        # Прогноз для layer_2 с применением функции-активатора softmax.
        # Форма layer_2_delta - (128, 10).
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        # Сравнение полученного прогноза выходного слоя layer_2 и фактического изображения цифры.
        # batch_size - количество разбитых групп изображений.
        for k in range(batch_size):
            labelset = labels[batch_start+k:batch_start+k+1]
            _inc = int(np.argmax(layer_2[k:k+1]) == np.argmax(labelset))
            correct_cnt += _inc

        # Разности вычисляются в обратном порядке с нюансами вычисления первой.
        # labels[batch_start:batch_end] и layer_2 имеют одинаковую форму (128, 10).
        # Форма layer_2_delta - (128, 10).
        # Форма weights_1_2.T - (10000, 10).
        # Форма layer_1_delta  - (128, 10000).
        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        # Обратное распространение с применением функции-активатора tanh2deriv и фильтрацией dropout_mask.
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask

        # Вычисление приращений и корректировка весов
        weights_1_2 += alpha * np.dot(layer_1.T, layer_2_delta)
        # Reshape из (128, 10000) = 128000 в (80000, 16) = 128000 для правильного векторного умножения.
        layer_1_delta_reshape = layer_1_delta.reshape(kernel_output.shape)
        # Форма kernels и kernels_update - (9, 16).
        kernels_update = np.dot(flattened_input.T, layer_1_delta_reshape)
        kernels += alpha * kernels_update

    # Определение точность нейронной сети на данных, не участвовавших в ее обучении.
    # Принцип вычисления остаются прежним с некоторым упрощением.
    test_correct_cnt = 0

    for i in range(len(test_images)):

        layer_0 = test_images[i:i+1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0,
                                         row_start,
                                         row_start+kernel_rows,
                                         col_start,
                                         col_start+kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = np.dot(flattened_input, kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        layer_2 = np.dot(layer_1, weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

    if j % 1 == 0:
        sys.stdout.write(f"\n Iteration: {j} Test accuracy: {test_correct_cnt/float(len(test_images))} "
                         f"Train accuracy: {correct_cnt/float(len(images))}")
