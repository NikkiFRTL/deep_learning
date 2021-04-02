import numpy as np


class Tensor:
    """
    C системой автоматического вычисления градиента autograd код
    становится намного проще. Нам больше не нужны временные переменные
    (потому что вся необходимая информация сохраняется в динамическом вычислительном
    графе) и нет необходимости вручную определять логику обратного
    распространения (потому что она уже реализована в методе . backward ()).
    """
    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):

        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}

        if id is None:
            self.id = np.random.randint(0, 100000)
        else:
            self.id = id

        if self.creators is not None:
            for creator in creators:
                if self.id not in creator.children:  # Keeps track of how many children a tensor has.
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        """
        Checks whether a tensor has received the correct number of gradients from each child
        """
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):

        if self.autograd:

            # Позволит не передавать градиент из единиц в первый вызов .backward().
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            # Accumulates gradients from several children
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            # Grads must not have grads of their own
            assert grad.autograd is False

            # 1) only continue backpropping if there's something to
            # backprop into and if all gradients (from children)
            # 2) are accounted for override waiting for children if
            # "backprop" was called on this variable directly
            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):

                # Begins actual backpropagation
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if self.creation_op == "mm":
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))

                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if self.creation_op == "index_select":
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def index_select(self, indices):
        """
        Сохраняет исходные индексы, чтобы на этапе обратного распространения мы смогли поместить каждый градиент
        в нужное место, воспользовавшись простым циклом for.
        """
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    Оптимизатор стохастического градиентного спуска
    """
    def __init__(self, parameters, alpha=0.01):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha

            if zero:
                p.grad.data *= 0


class Layer:
    """
    Абстракция слоя.
    Это коллекция процедур, часто используемых в прямом распространении, упакованных в простой программный
    интерфейс с методом .forward() для их использования.
    """
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        """
        Этот метод можно использовать для определения более сложных типов слоев
        (например, слоев, содержащих другие слои).
        Останется только переопределить метод get_parameters(), чтобы организовать передачу нужных тензоров.
        """
        return self.parameters


class Linear(Layer):
    """
    Модель линейного слоя.
    """

    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        # Xavier initialization of Normal distribution (random.randn).
        # np.random.rand is for Uniform distribution (in the half-open interval [0.0, 1.0))
        # np.random.randn is for Standard Normal (Gaussian) distribution (mean 0 and variance 1)
        weight = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(weight, autograd=True)

        # Матрица смещений весов, потому что это настоящий линейный слой.
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    """
    Модель последовательного слоя, который осуществляет прямое распространение через список слоев,
    когда выход одного слоя передается на вход следующего.
    """
    def __init__(self, layers=list()):
        super().__init__()

        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for layer in self.layers:
            params += layer.get_parameters()
        return params


class MSELoss(Layer):
    """
    Loss-function layer - Mean Squared Error.
    Слой с функцией потерь - среднеквадратическая ошибка.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        return ((prediction - target) * (prediction - target)).sum(0)


class Tanh(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()


class Embedding(Layer):
    """
    Слой векторного представления, который изучает векторные представления слов.
    Слой должен инициализировать список (правильной длины) векторных представлений слов (правильного размера).

    """
    def __init__(self, vocab_size, dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        # This random initialisation style is just a convention from word2vec
        # Матрица содержит по одной строке (вектору) для каждого слова из словаря
        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)
