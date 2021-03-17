import sys, random, math
import numpy as np
from collections import Counter


np.random.seed(1)
random.seed(1)

with open("reviews.txt") as f:
    raw_reviews = f.readlines()

tokens = [x.split(" ") for x in raw_reviews]

word_counter = Counter()

for sentence in tokens:
    for word in sentence:
        word_counter[word] -= 1

vocab = list(set(map(lambda x: x[0], word_counter.most_common())))

word2index = dict()

for index, word in enumerate(vocab):
    word2index[word] = index

concatenated = list()
input_dataset = list()
for sentence in tokens:
    sentence_indices = list()
    for word in sentence:
        try:
            sentence_indices.append(word2index[word])
            concatenated.append(word2index[word])
        except:
            ""
    input_dataset.append(sentence_indices)

concatenated = np.array(concatenated)

random.shuffle(input_dataset)

alpha = 0.02
iterations = 2
hidden_size = 50
window = 2
negative = 5

# random.rand(x, y) - случайные числа [0:1) в заданной форме матрицы х, у.
weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0


layer_2_target = np.zeros(negative+1)
layer_2_target[0] = 1


def similar(target='beautiful'):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for review_index, review in enumerate(input_dataset * iterations):
    for target_index in range(len(review)):

        # Прогнозировать только случайное подмножество,
        # потому что прогнозирование всего словаря обходится слишком дорого.

        # np.random.rand(negative) например = [0.212, 0.234, 0.324, 0.657, 0.934].
        # умножении вектора на число len(concatenated) = N. [0.212*N, 0.234*N, 0.324*N, 0.657*N, 0.934*N].
        # .astype('int') - форматирование чисел в int.
        # .tolist() - форматирование вектора в список цифр.
        # concatenated[....] - i-тые значения в матрице concatenated
        # target_samples равен сумме 2х списков, состоящий их 1 индекса слова и 5 случайных индексов слов из review.
        target_samples = [review[target_index]]+list(concatenated[(np.random.rand(negative) *
                                                                   len(concatenated)).astype('int').tolist()])

        # Список из двух индексов слов из review.
        left_context = review[max(0, target_index-window):target_index]
        # Список из одного индекса слова из review.
        right_context = review[target_index+1:min(len(review), target_index+window)]

        # mean() вычисляет среднее арифметическое значений элементов массива.
        # TODO from here
        layer_1 = np.mean(weights_0_1[left_context+right_context], axis=0)
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2[target_samples].T))

        layer_2_delta = layer_2 - layer_2_target
        layer_1_delta = np.dot(layer_2_delta, weights_1_2[target_samples])

        weights_0_1[left_context+right_context] -= alpha * layer_1_delta
        weights_1_2[target_samples] -= alpha * np.outer(layer_2_delta, layer_1)

    if review_index % 250 == 0:
        pass
        #sys.stdout.write(f"\rProgress: {review_index/float(len(input_dataset) * iterations)}  {similar('doubt')}")
    #sys.stdout.write(f"\rProgress: {review_index / float(len(input_dataset) * iterations)}")

#print(similar('doubt'))
