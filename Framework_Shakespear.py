import sys, random, math
from collections import Counter
import numpy as np
from Framework_Code import Tensor, SGD, MSELoss, Tanh, Sigmoid, Embedding, CrossEntropyLoss, RNNCell, LSTMCell

np.random.seed(0)

with open('tasksv11/shakespear.txt') as f:
    raw = f.read()

# Словарь с индексами каждого символа из текста shakespear.
vocab = list(set(raw))
word2index = {}
for index, word in enumerate(vocab):
    word2index[word] = index
indices = np.array([word2index[x] for x in raw])

# Этот код инициализирует векторные представления с размерностью 8 и рекуррентную нейронную сеть с размерностью
# скрытого состояния 512.
# Выходные веса инициализируются нулями (не является обязательным требованием, но сеть работает немного лучше).
# В конце инициализируется оптимизатор стохастического градиентного спуска с перекрестной энтропией
# в качестве функции потерь.

embed = Embedding(vocab_size=len(vocab), dim=512)

model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))

# Небольшая помощь при обучении.
model.w_ho.weight.data *= 0

criterion = CrossEntropyLoss()

weight_optimizer = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)

"""
Выполнять обратное распространение через 100 000 символов очень непрактично, т.к. это слишком много,
чтобы выполнять его для каждого прогноза.
Выход - выполнить обратное распространение на определенное число шагов и остановиться!
Этот прием называется УСЕЧЕННЫМ ОБРАТНЫМ РАСПРОСТРАНЕНИЕМ.
Количество шагов обратного распространения в этом случае становится еще одним настраиваемым параметром.

Недостатком использования усеченного обратного распространения является сокращение расстояния,
которое нейронная сеть может запомнить.
Сеть не сможет научиться делать прогнозы, опираясь на входной сигнал, отстоящий далее чем на N шагов в прошлом.

Другой недостаток усеченного обратного распространения — усложнение логики формирования пакетов. 
Чтобы использовать усеченное обратное распространение, мы должны сделать вид, что у нас не один большой набор данных,
а много маленьких наборов, размер каждого из которых равен bptt.
"""
# Переменная bptt, определяющая границу усечения (обычно 16 - 64).
bptt = 16
batch_size = 25
n_batches = int(indices.shape[0] / batch_size)
n_bptt = int((n_batches - 1) / bptt)

# Группировка наборов данных.
# Усекаем набор данных до размера, кратного произведению batch_size и n_batches,
# чтобы привести его к прямоугольной форме перед группировкой в тензоры.
trimmed_indices = indices[:n_batches * batch_size]
# Изменяем форму набора данных так, чтобы каждый столбец представлял сегмент начального массива индексов.
batched_indices = trimmed_indices.reshape(batch_size, n_batches).T

# A = [0, 1, 2, 3]
input_batched_indices = batched_indices[0:-1]  # A[0:-1] >>> [0, 1, 2]
# Целевые индексы — это входные индексы со смещением на одну строку (сеть обучается предсказывать следующий символ).
target_batched_indices = batched_indices[1:]  # A[1:] >>> [1, 2, 3]

input_batches = input_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
target_batches = target_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)

"""
Следующий код демонстрирует практическую реализацию усеченного обратного распространения.
batch_loss генерируется на каждом шаге; и после каждых bptt шагов выполняется обратное
распространение и корректировка весов. 
Затем чтение данных продолжается, как если бы ничего не произошло (и с использованием того же скрытого состояния,
что и прежде, которое сбрасывается только при смене эпохи)
"""


def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        output.data *= 15
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()
        m = output.data.argmax()  # Выбрать в качестве прогноза символы с максимальной вероятностью.
        с = vocab[m]
        input = Tensor(np.array([m]))
        s += с
    return s


def train(iterations=100):

    for iter in range(iterations):
        total_loss = 0
        n_loss = 0

        hidden = model.init_hidden(batch_size=batch_size)

        for batch_i in range(len(input_batches)):
            #
            hidden = (Tensor(hidden[0].data, autograd=True),
                      Tensor(hidden[1].data, autograd=True))
            loss = None
            losses = list()

            for t in range(bptt):
                input = Tensor(input_batches[batch_i][t], autograd=True)
                rnn_input = embed.forward(input=input)
                output, hidden = model.forward(input=rnn_input, hidden=hidden)

                target = Tensor(target_batches[batch_i][t], autograd=True)
                batch_loss = criterion.forward(output, target)

                if t == 0:
                    losses.append(batch_loss)
                else:
                    losses.append(batch_loss + losses[-1])
            loss = losses[-1]

            loss.backward()

            weight_optimizer.step()

            total_loss += loss.data / bptt
            epoch_loss = np.exp(total_loss / (batch_i+1))

            log = f"\rIter: {str(iter)} - Alpha: {str(weight_optimizer.alpha)[0:5]} " \
                  f"- Batch {str(batch_i + 1)}/{str(len(input_batches))} " \
                  f"- Min Loss: {str(epoch_loss)[0:5]} " \
                  f"- Loss: {str(epoch_loss)}"
            if batch_i == 0:
                s = generate_sample(n=70, init_char='T').replace("\n", " ")
                log += "-" + s
            if batch_i % 1 == 0:
                sys.stdout.write(log)
        weight_optimizer.alpha *= 0.99


train(100)

# print(generate_sample(n=500, init_char='\n'))
