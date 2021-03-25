import numpy as np


with open('tasksv11/en/qa1_single-supporting-fact_train.txt') as f:
    raw_text = f.readlines()

tokens = list()
for line in raw_text[0:1000]:
    tokens.append(line.lower().replace('\n', '').split(' ')[1:])

vocab = set()
for sentence in tokens:
    for word in sentence:
        vocab.add(word)
vocab = list(vocab)

word2index = {}
for index, word in enumerate(vocab):
    word2index[word] = index


def words2indices(sentence):
    indices = list()
    for word in sentence:
        indices.append(word2index[word])
    return indices


def softmax(x):
    e_x = (x - np.max(x))
    return e_x / (e_x.sum(axis=0))


np.random.seed(1)
embed_size = 10

embed = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1
recurrent = np.eye(embed_size)
start = np.zeros(embed_size)
decoder = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1
one_hot = np.eye(len(vocab))


def predict(sentence):

    layers = list()
    layer = {}
    layer['hidden'] = start
    layers.append(layer)

    loss = 0

    for target_index in range(len(sentence)):

        layer = {}
        layer['pred'] = softmax(np.dot(layers[-1]['hidden'], decoder))
        loss += -np.log(layer['pred'][sentence[target_index]])
        layer['hidden'] = np.dot(layers[-1]['hidden'], recurrent) + embed(sentence[target_index])
        layers.append(layer)

    return layers, loss


for iteration in range(30):

    sentence = words2indices(tokens[iteration % len(tokens)][1:])
    layers, loss = predict(sentence)
    alpha = 0.01 / float(len(sentence))

    for layer_index in range(reversed(len(layers))):

        layer = layers[layer_index]
        target = sentence[layer_index - 1]

        if layer_index > 0:
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = np.dot(layer['output_delta'], decoder.T)

            if layer_index == len(layers) - 1:
                layer['hidden_delta'] = new_hidden_delta

            else:
                layer['hidden_delta'] = new_hidden_delta + np.dot(layers[layer_index+1]['hidden_delta'], recurrent.T)

        else:
            layer['hidden_delta'] = np.dot(layers[layer_index+1]['hidden_delta'], recurrent.T)

    start -= layers[0]['hidden_delta'] * alpha

    for layer_index, layer in enumerate(layers[1:]):

        decoder -= np.outer(layers[layer_index]['hidden'], layer['output_delta']) * alpha

        embed_index = sentence[layer_index]
        embed[embed_index] -= layers[layer_index]['hidden_delta'] * alpha

        recurrent -= np.outer(layers[layer_index]['hidden'], layer['hidden_delta']) * alpha

    if iteration % 10 == 0:
        print(f"Perplexity :  {np.exp(loss/float(len(sentence)))}")

sent_index = 4

all_layers, _ = predict(words2indices(tokens[sent_index]))

print(tokens[sent_index])

for index, layer in enumerate(all_layers[1:-1]):
    input = tokens[sent_index][index]
    true = tokens[sent_index][index+1]
    prediction = vocab[layer['pred'].argmax()]
    print(input, true, prediction)
