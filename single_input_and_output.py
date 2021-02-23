"""
Реализация градиентного спуска с одним входом
"""
games = [1.1]
expectation = 0.8
weight = 0.0
alpha = 0.01
input = games[0]

# Процесс обучения из 4х итераций.
for iteration in range(4):
    print(f"\nWeight: {weight}")
    prediction = input * weight
    error = (prediction - expectation) ** 2
    delta = prediction - expectation
    weight_delta = delta * input
    weight = weight - weight_delta
    print(f"Error: {error}! Prediction: {prediction}")
    print(f"Delta: {delta} Weight_delta: {weight_delta}")
