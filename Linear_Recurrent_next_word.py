import numpy as np


# Файл с данными.
with open('tasksv11/en/qa1_single-supporting-fact_train.txt', 'r') as f:
    raw = f.readlines()

# Разделение текста на токены.
tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\n", "").split(" ")[1:])


# ПОДГОТОВКА И ОПРЕДЕЛЕНИЕ ПАРАМЕТОВ

# Список неповоторяющихся слов. Длина - 78.
vocab = set()
for sentence in tokens:
    for word in sentence:
        vocab.add(word)
vocab = list(vocab)

# Словарь "слово - индекс".
word2index = {}
for index, word in enumerate(vocab):
    word2index[word] = index


def words2indices(sentence):
    """
    Функция для преобразования списка слов в список индексов.
    """
    indices = list()
    for word in sentence:
        indices.append(word2index[word])
    return indices


def softmax(x):
    """
    Функция активации softmax для предсказания следующего слова.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# Инициализирует начальное значение случайной последовательности.
np.random.seed(1)

# Размер векторного представления.
embed_size = 10

# Матрица векторных представлений слов.
# Размер (78, 10)
embed = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1

# Рекуррентная матрица (первоначально единичная).
# Размер (10, 10)
recurrent = np.eye(embed_size)

# Начальное векторное представление предложения (соответствует пустой фразе).
# Размер (10,)
start = np.zeros(embed_size)

# Весовая матрица для прогнозирования векторного представления предложения.
# Размер (78, 10)
decoder = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1

# Вспомогательная матрица. Матрица поиска выходных весов (для функции потерь).
# Размер (78, 78)
one_hot = np.eye(len(vocab))

# ПРЯМОЕ РАСПРОСТРАНЕНИЕ С ДАННЫМИ ПРОИЗВОЛЬНОЙ ДЛИНЫ (Forward propagation with arbitrary length)


def predict(sentence):
    """
    Следующий код реализует логику прямого распространения и предсказания следующего слова.
    Вместо предсказания только по последнему слову мы вычисляем предсказание (layer['pred']) на каждом шаге,
    опираясь на векторное представление, сгенерированное из предыдущих слов.
    """
    # Список с именем layers — этo способ организации прямого распространения.
    layers = list()
    # Из-за невозможности использовать статические переменные для хранения слоев, как данными фиксированной длины,
    # нужно добавлять новые слои в список, в зависимости от требуемого их количества.
    layer = {}
    # Скрытый слой вектора представления предложения.
    layer['hidden'] = start
    layers.append(layer)

    loss = 0

    for target_index in range(len(sentence)):

        layer = {}

        # Попытка предсказать следующее слово.
        layer['pred'] = softmax(np.dot(layers[-1]['hidden'], decoder))

        # Перплексия.
        # — это результат извлечения логарифма из вероятности получения правильной метки (слова),
        # последующего отрицания и вычисления экспоненты (е**х).
        # Теоретически она представляет разность между двумя распределениями вероятностей.
        # В данном случае идеальным считается такое распределение вероятностей,
        # когда правильное слово получает 100%-ную вероятность, а все остальные 0%.
        # Перплексия получает большое значение, когда два распределения вероятностей не совпадают,
        # и низкое (близкое к 1), когда совпадают.
        loss += -np.log(layer['pred'][sentence[target_index]])

        # Сгенерировать следующее состояние скрытого слоя.
        layer['hidden'] = np.dot(layers[-1]['hidden'], recurrent) + embed[sentence[target_index]]

        layers.append(layer)

    return layers, loss


# ОБРАТНОЕ РАСПРОСТРАНЕНИЕ С ДАННЫМИ ПРОИЗВОЛЬНОЙ ДЛИНЫ (Backpropagation with arbitrary length)

# Часть прямого распространения.
for iteration in range(2900):
    alpha = 0.01
    sentence = words2indices(tokens[iteration % len(tokens)][1:])
    # Получение списка со слоями слов предложения и перплексии.
    layers, loss = predict(sentence)

# Часть обратного распространения.
    for layer_index in reversed(range(len(layers))):
        # Извлечение слоев из списка.
        layer = layers[layer_index]
        # Опоределение искомого слова. В данном случае предшествующее.
        target = sentence[layer_index - 1]

        # В каждом слое layer есть 'hidden' и 'pred', добавим еще 'hidden_delta' и 'output_delta',
        # представляющие градиент в каждом их слоев.

        # Если это не первый слой.
        if layer_index > 0:
            # Определение градиента выходного слоя как разница веса предсказания и веса целевого слова.
            layer['output_delta'] = layer['pred'] - one_hot[target]
            # Вычисление обновленного градиента.
            new_hidden_delta = np.dot(layer['output_delta'], decoder.T)

            # Если это последний слой, то не брать данные из следующего, потому что его не существует.
            if layer_index == len(layers)-1:
                layer['hidden_delta'] = new_hidden_delta
            # Если слой не последний.
            else:
                layer['hidden_delta'] = np.dot(layers[layer_index+1]['hidden_delta'], recurrent.T) + new_hidden_delta

        # Если это первый слой.
        else:
            layer['hidden_delta'] = np.dot(layers[layer_index+1]['hidden_delta'], recurrent.T)

# КОРРЕКТИРОВКА ВЕСОВ С ДАННЫМИ ПРОИЗВОЛЬНОЙ ДЛИНЫ (Weight update with arbitrary length)

    # Обновление весов.
    start -= layers[0]['hidden_delta'] * alpha / float(len(sentence))

    for layer_index, layer in enumerate(layers[1:]):

        decoder -= np.outer(layers[layer_index]['hidden'], layer['output_delta']) * alpha / float(len(sentence))

        embed_index = sentence[layer_index]
        embed[embed_index] -= layers[layer_index]['hidden_delta'] * alpha/float(len(sentence))

        recurrent -= np.outer(layers[layer_index]['hidden'], layer['hidden_delta']) * alpha / float(len(sentence))

    if iteration % 1 == 0:
        print(f"Perplexity:  {np.exp(loss / len(sentence))}")

sent_index = 4

l, _ = predict(words2indices(tokens[sent_index]))

print(tokens[sent_index])

for i, each_layer in enumerate(l[1:-1]):
    input = tokens[sent_index][i]
    true = tokens[sent_index][i+1]
    pred = vocab[each_layer['pred'].argmax()]
    print(f"Prev Input: {input} {' ' * (12 - len(input))}  True: {true}{' ' * (15 - len(true))}   Pred:  {pred}")
