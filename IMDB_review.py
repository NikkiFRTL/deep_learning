import numpy as np
import sys
from collections import Counter
import math
np.random.seed(1)


def similar(target='great'):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


with open("Harry Potter and the sorcerer's stone.txt") as f:
    raw_reviews = f.readlines()

with open('labels.txt') as f:
    raw_labels = f.readlines()

tokens = [set(x.split(" ")) for x in raw_reviews]

vocab = set()
for sentence in tokens:
    for word in sentence:
        if len(word) > 0:
            vocab.add(word)
vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

input_dataset = list()
for sentence in tokens:
    sentence_indices = list()
    for word in sentence:
        try:
            sentence_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sentence_indices)))


target_dataset = list()
for label in raw_labels:
    if label == 'positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)


alpha = 0.01
iterations = 10
hidden_size = 2

weights_0_1 = 0.2*np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size, 1)) - 0.1

correct = 0
total = 0
print(len(target_dataset))
print(len(input_dataset))
for iter in range(iterations):
    for i in range(len(input_dataset)):
        x = input_dataset[i]
        y = target_dataset[i]
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        layer_2_delta = layer_2 - y
        layer_1_delta = np.dot(layer_2_delta, weights_1_2.T)

        weights_0_1[x] -= alpha * layer_1_delta
        weights_1_2 -= alpha * np.outer(layer_1, layer_2_delta)

        if np.abs(layer_2_delta) < 0.5:
            correct += 1
        total += 1

        if i % 10 == 9:
            progress = f"{i/float(len(input_dataset))}"
            sys.stdout.write(f"\r Iteration: {iter} Progress: {progress[2:4]}.{progress[4:6]} "
                             f"Training Accuracy: {correct/float(total)}")

    print()

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

print(similar())
