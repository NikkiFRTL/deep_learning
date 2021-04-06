from Framework_Code import Tensor, SGD, Sequential, Linear, MSELoss, Tanh, Sigmoid, Embedding, CrossEntropyLoss
import numpy as np
"""
Реализовав эту нейронную сеть, мы выявили корреляцию между входными индексами 1 и 2 и прогнозами 0 и 1. 
Теоретически индексы 1 и 2 могут соответствовать словам (или некоторым другим входным объектам), и в заключительном
примере мы выявим это соответствие. Этот пример должен был показать, как действует слой с векторным представлением.
"""

np.random.seed(0)
# Исходные индексы.
data = Tensor(np.array([1, 2, 1, 2]), autograd=True)

# Целевые индексы.
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

embed = Embedding(3, 3)

# Список весов(слоёв) запакованный в класс.
# embed, Tanh(), Linear(3, 1), Sigmoid() - для реализации среднеквадратичной ошибки MSELoss.
model = Sequential([embed, Tanh(), Linear(3, 4)])

# Слой с функцией потерь - среднеквадратическая ошибка.
# criterion = MSELoss() - для реализации среднеквадратичной ошибки MSELoss.
# criterion = CrossEntropyLoss() - для реализации перекрестной энтропии (для ошибки).
criterion = CrossEntropyLoss()

# Корректировка весов (оптимизатор стохастического градиентного спуска).
weights_optimizer = SGD(parameters=model.get_parameters(), alpha=0.1)

for i in range(10):

    # Прогноз (векторное умножение layer_0 на weights_0_1, а результат на weights_1_2).
    # Преобраузем pred = data.mm(weights[0]).mm(weights[1])
    pred = model.forward(data)

    # Потеря как среднеквадратичная ошибка.
    loss = criterion.forward(pred, target)

    # Обратное распространение.
    loss.backward(Tensor(np.ones_like(loss.data)))
    # Корректировака весов.
    weights_optimizer.step()

    print(loss)

