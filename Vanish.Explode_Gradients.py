import numpy as np

"""
Ниже показан цикл обратного распространения в рекуррентной нейронной сети с функциями активации sigmoid и relu. 
Обратите внимание, как градиенты становятся очень маленькими/болыпими для sigmoid/relu соответственно.
Увеличение обусловлено матричным умножением, а уменьшение объясняется плоской производной в хвостах 
функции sigmoid (это характерно для многих нелинейных функций активации).

Возникает эффект затухания или взрывного роста градиентов при изменении скрытых состояний в рекуррентной нейронной сети. 
Проблема заключается в использовании комбинации матричного умножения и нелинейной функции активации 
для формирования скрытого состояния.

Решением является Модель LSTM (Ячейки долгой краткосрочной памяти).
"""

sigmoid = lambda x: 1/(1+np.exp(-x))
relu = lambda x: (x > 0).astype(float)*x

weights = np.array([[1, 4], [4, 1]])

activation = sigmoid(np.array([1, 0.01]))

print("\nSigmoid Activations")

activations = list()
for iter in range(5):
    activation = sigmoid(activation.dot(weights))
    activations.append(activation)
    print(activation)

print("\nSigmoid Gradients")

gradient = np.ones_like(activation)
for activation in reversed(activations):
    # Производная функции sigmoid способствует чрезмерному уменьшению градиентов,
    # когда ее значение близко к 0 или 1 (хвосты).
    gradient = (activation * (1 - activation) * gradient)
    gradient = gradient.dot(weights.transpose())
    print(gradient)

print("\nRelu Activations")

activations = list()
for iter in range(5):
    # Матричное умножение вызывает взрывной рост градиентов, который не компенсируется нелинейной функцией активации
    # (такой, как sigmoid)
    activation = relu(activation.dot(weights))
    activations.append(activation)
    print(activation)

print("\nRelu Gradients")

gradient = np.ones_like(activation)
for activation in reversed(activations):
    gradient = ((activation > 0) * gradient)
    gradient = gradient.dot(weights.transpose())
    print(gradient)
