import numpy as np
from collections import Counter
import random
import sys
import codecs
import copy
from Framework_Code import Tensor, SGD, MSELoss, Tanh, Sigmoid, Embedding, CrossEntropyLoss, RNNCell, LSTMCell
import phe

np.random.seed(12345)

# ПОДГОТОВКА ДАННЫХ
with open('tasksv11/spam.txt', "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

vocab, spam, ham = (set(["<unk>"]), list(), list())
for row in raw:
    spam.append(set(row[:-2].split(" ")))
    for word in spam[-1]:
        vocab.add(word)

with codecs.open('tasksv11/ham.txt', "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

for row in raw:
    ham.append(set(row[:-2].split(" ")))
    for word in ham[-1]:
        vocab.add(word)

vocab = list(vocab)
word2index ={}
for index, word in enumerate(vocab):
    word2index[word] = index


def to_indices(input, length=500):
    indices = list()
    for line in input:
        if len(line) < length:
            # Дополняющая до 500 знаков лексема.
            line = list(line) + ["<unk>"] * (length - len(line))
            idxs = list()
            for word in line:
                idxs.append(word2index[word])
            indices.append(idxs)
    return indices


# ОБУЧЕНИЕ ВЫЯВЛЯТЬ СПАМ
spam_idx = to_indices(spam)
ham_idx = to_indices(ham)

train_spam_idx = spam_idx[0:-1000]
train_ham_idx = ham_idx[0:-1000]

test_spam_idx = spam_idx[-1000:]
test_ham_idx = ham_idx[-1000:]

train_data = list()
train_target = list()

test_data = list()
test_target = list()

for i in range(max(len(train_spam_idx), len(ham_idx))):
    train_data.append(train_spam_idx[i % len(train_spam_idx)])
    train_target.append([1])
    train_data.append(train_ham_idx[i % len(train_spam_idx)])
    train_target.append([0])

for i in range(max(len(test_spam_idx), len(test_ham_idx))):
    test_data.append(test_spam_idx[i % len(test_spam_idx)])
    test_target.append([1])
    test_data.append(test_ham_idx[i % len(test_ham_idx)])
    test_target.append([0])


def train(model, input_data, target_data, batch_size=500, iterations=5):
    n_batches = int(len(input_data) / batch_size)
    for iter in range(iterations):
        iter_loss = 0
        for b_i in range(n_batches):
            # Дополняющая лексема не должна оказывать влияния на прогноз.
            model.weight.data[word2index["<unk>"]] *= 0
            input = Tensor(input_data[b_i * batch_size:(b_i+1) * batch_size], autograd=True)
            target = Tensor(target_data[b_i * batch_size:(b_i+1) * batch_size], autograd=True)

            pred = model.forward(input).sum(1).sigmoid()
            loss = criterion.forward(pred, target)
            loss.backward()
            weight_optimizer.step()

            iter_loss += loss.data[0] / batch_size

            sys.stdout.write(f"\r\tLoss: {iter_loss / (b_i+1)}")
        print()
    return model


def test(model, test_input, test_output):

    model.weight.data[word2index['<unk>']] *= 0

    input = Tensor(test_input, autograd=True)
    target = Tensor(test_output, autograd=True)

    pred = model.forward(input).sum(1).sigmoid()
    return ((pred.data > 0.5) == target.data).mean()


model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0
criterion = MSELoss()
weight_optimizer = SGD(parameters=model.get_parameters(), alpha=0.01)

for i in range(1):  # Не федеретивная модель - без защиты данных пользователей.
    model = train(model, train_data, train_target, iterations=1)
    print(f"Correct on Test Set: {test(model, test_data, test_target) * 100}")

# Сделаем модель федеративной.

bob = train_data[0:1000], train_target[0:1000]
alice = train_data[1000:2000], train_target[1000:2000]
sue = train_data[2000:], train_target[2000:]

model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0
# При использовании этого алгоритма в действующем программном обеспечении значение n_length должно быть не меньше 1024!
public_key, private_key = phe.generate_paillier_keypair(n_length=128)


def train_and_encrypt(model, input, target, pubkey):
    new_model = train(copy.deepcopy(model), input, target, iterations=1)

    encrypted_weights = list()
    for val in new_model.weight.data[:, 0]:
        encrypted_weights.append(public_key.encrypt(val))
    ew = np.array(encrypted_weights).reshape(new_model.weight.data.shape)
    return ew


for i in range(3):
    print(f"Starting Training Round...")
    print(f"\tStep 1: Send the model to Bob")
    bob_encrypted_model = train_and_encrypt(copy.deepcopy(model), bob[0], bob[1], public_key)

    print(f"\n\tStep 2: Send the model to ALice")
    alice_encrypted_model = train_and_encrypt(copy.deepcopy(model), alice[0], alice[1], public_key)

    print(f"\n\tStep 3: Send the model to Sue")
    sue_encrypted_model = train_and_encrypt(copy.deepcopy(model), sue[0], sue[1], public_key)

    print("\n\tStep 4: Bob, Alice, and Sue send their")
    print("\tencrypted models to each other.")
    aggregated_model = bob_encrypted_model + alice_encrypted_model + sue_encrypted_model

    print("\n\tStep 5: only the aggregated model")
    print("\tis sent back to the model owner who")
    print("\t can decrypt it.")
    raw_values = list()
    for val in aggregated_model.flatten():
        raw_values.append(private_key.decrypt(val))
    new = np.array(raw_values).reshape(model.weight.data.shape) / 3
    model.weight.data = new

    print(f"\tCorrect on Test Set: {test(model, test_data, test_target) * 100}")
