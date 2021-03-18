import numpy as np
import sys
from collections import Counter
import math
np.random.seed(1)


def similar(target='great'):
    """
    Функция нахождения
    """
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)


def sigmoid(x):
    """
    Используется здесь т.к. она способна плавно сжимать бескорнечный диапозон входных значений в диапозон от 0 до 1.
    Это позволяет интерпритировать выход любго отдельного нейрона как вероятность.
    """
    return 1/(1 + np.exp(-x))


with open("Harry Potter and the sorcerer's stone.txt") as f:
    raw_reviews = f.readlines()

with open('labels.txt') as f:
    raw_labels = f.readlines()

# Генератор списка предложений, преобразуемый в сет не повторяющихся слов.
tokens = [set(x.split(" ")) for x in raw_reviews]

# Создание списка с сетом слов и фильтрацией > 0.
vocab = set()
for sentence in tokens:
    for word in sentence:
        if len(word) > 0:
            vocab.add(word)

vocab = list(vocab)


# Создание словаря ключ-значение в ввиде слово-цифра.
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

# Подготовка данных для передачи в сеть.
input_dataset = list()
for sentence in tokens:
    sentence_indices = list()
    # Перебираем слова в предложении и добавляем в список sentence_indices значения соответствующие ключу
    # этого слова в словаре word2index.
    for word in sentence:
        try:
            sentence_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sentence_indices)))

# Данные ожидаемой (известной) оценки ревью (positive=1 / negative=0).
target_dataset = list()
for label in raw_labels:
    if label == 'positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)


alpha = 0.01
iterations = 1
hidden_size = 2

weights_0_1 = 0.2*np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size, 1)) - 0.1

correct = 0
total = 0

# Предсказание и корректировка весов.
for iter in range(iterations):

    # len(input_dataset) - это количество слов(их цифровых представлений) в одном обзоре.
    for i in range(len(input_dataset)):

        x = input_dataset[i]
        y = target_dataset[i]

        # Векторное представление (используются веса только присутствующих в предложении слов) + sigmoid.
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        # Разность между прогнозом и истиной для предсказания эмоциональной окраски.
        layer_2_delta = layer_2 - y
        # Обратное распространение.
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T)

        # Функция outer() вычисляет внешнее произведение двух векторов.
        # Каждого числа из вектора А на каждое число вектора B без суммирования. Результат - матрица.
        weights_0_1[x] -= alpha * layer_1_delta
        weights_1_2 -= alpha * np.outer(layer_1, layer_2_delta)

        # Проверка корреляции, чем ближе к 0.0, тем ближе по значению к 'positive'.
        if np.abs(layer_2_delta) < 0.5:
            correct += 1
        total += 1

        if i % 10 == 0:
            progress = f"{i/float(len(input_dataset))}"
            sys.stdout.write(f"\r Iteration: {iter} Progress: {progress[2:4]}.{progress[4:6]} "
                             f"Training Accuracy: {correct/float(total)}")

    print()

# Проверка точности прогнозирования на данных, не участвующих в обучении.
correct = 0
total = 0
for i in range(len(input_dataset)):
    x = input_dataset[i]
    y = target_dataset[i]
    layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

    if np.abs(layer_2 - y) < 0.5:
        correct += 1
    total += 1

print(f"Test Accuracy: {correct/float(total)}")
