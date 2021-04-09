import sys, random, math
from collections import Counter
import numpy as np
from Framework_Code import Tensor, SGD, Sequential, Linear, MSELoss, Tanh, Sigmoid, Embedding, CrossEntropyLoss, RNNCell


with open('tasksv11/en/qa1_single-supporting-fact_train.txt') as f:
    raw = f.readlines()

tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\n", "").split(" ")[1:])

new_tokens = list()
for line in tokens:
    new_tokens.append(['-'] * (6 - len(line)) + line)

tokens = new_tokens

vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {}
for index, word in enumerate(vocab):
    word2index[word] = index


def word2indices(sentence):
    idx = list()
    for w in sentence:
        idx.append(word2index[w])
    return idx


indices = list()
for line in tokens:
    indices.append(word2indices(line))

# Теперь можно инициализировать рекуррентный слой входными векторными представлениями и обучить сеть.
# Эта сеть немного сложнее (она имеет один дополнительный слой).
data = np.array(indices)

# Здесь сначала определяются входные векторные представления, а затем инициализируется рекуррентная ячейка.
# Eдиничный рекуррентный слой принято называть ячейкой.
# Если создать другой слой, объединяющий произвольное число ячеек, он будет называться рекуррентной
# нейронной сетью и принимать дополнительный входной параметр n_layers.

embed = Embedding(vocab_size=len(vocab), dim=16)
model = RNNCell(n_inputs=16, n_hidden=16, n_output=len(vocab))

criterion = CrossEntropyLoss()
params = model.get_parameters() + embed.get_parameters()
weight_optimizer = SGD(parameters=params, alpha=0.05)

for iter in range(1000):

    # Число пакетов = 100.
    batch_size = 100
    total_loss = 0

    # Инициализация скрытого слоя рекуррентной сети.
    hidden = model.init_hidden(batch_size=batch_size)

    # Объединение входных данных в пакеты с помощью вложенного цикла for.
    # В этом наборе данных нет ни одного предложения длиннее (6) шести слов.
    # Поэтому сеть читает пять первых слов и пыталается предсказать (6) шестое.
    for t in range(5):
        input = Tensor(data[0:batch_size, t], autograd=True)
        rnn_input = embed.forward(input=input)
        output, hidden = model.forward(rnn_input, hidden=hidden)

    # Целевое слово.
    target = Tensor(data[0:batch_size, t+1], autograd=True)

    # Прямое распространение.
    # Мы двигаем вперед пять входных примеров.
    loss = criterion.forward(output, target)

    # Обратное распространение.
    # Когда происходит вызов loss. backward(),
    # градиенты распространяются в обратном направлении через всю сеть до входных данных.
    loss.backward()

    # Обучение и корректировака весов.
    weight_optimizer.step()
    total_loss += loss.data

    if iter % 100 == 0:
        p_correct = (target.data == np.argmax(output.data, axis=1)).mean()
        print_loss = total_loss / (len(data)/batch_size)
        print(f"Loss: {print_loss}    Correct:  {p_correct}")

# Нейронная сеть, обученная на 100 первых примерах из обучающего набора данных,
# достигла точности прогнозирования около 37 % (почти идеально для этой задачи).
# Она предсказывает, куда может выйти Маrу.

# Попробуем предстаказть куда же пошла Mary напустив обученную сеть на пакет из одного предложения.
batch_size = 1
hidden = model.init_hidden(batch_size=batch_size)
for t in range(5):
    input = Tensor(data[0:batch_size, t], autograd=True)
    rnn_input = embed.forward(input=input)
    output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = Tensor(data[0:batch_size, t+1], autograd=True)
loss = criterion.forward(output, target)

context = ""
for idx in data[0:batch_size][0][0:-1]:
    context += vocab[idx] + " "
print("Context:", context)
print("True:", vocab[target.data[0]])
print("Pred:", vocab[output.data.argmax()])
