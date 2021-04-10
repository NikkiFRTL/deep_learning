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
                    return
                    print(self.id)
                    print(self.creation_op)
                    print(len(self.creators))
                    for c in self.creators:
                        print(c.creation_op)
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

                    # Инициализируем новый градиент правильного размера соответствующего размеру исходной матрицы,
                    # подвергавшейся индексированию).
                    new_grad = np.zeros_like(self.creators[0].data)

                    # Преобразуем индексы в плоский вектор, чтобы получить возможность выполнить итерации по ним.
                    indices_ = self.index_select_indices.data.flatten()

                    # Свернем grad_ в простой список строк.
                    # Индексы в indices_ и список векторов в grad_ будут иметь соответствующий порядок.
                    grad_ = grad.data.reshape(len(indices_), -1)

                    # Выполним обход всех индексов, добавим их в правильные строки в новой матрице градиентов
                    # и передадим ее обратно в self.creators[0].
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if self.creation_op == "cross_entropy":  # TODO describe this

                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

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
                          creation_op="expand_" + str(dim))
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
        Мы должны гарантировать на этапе обратного распространения размещение градиентов в тех же строках,
        полученных в результате индексирования на этапе прямого распространения.
        Для этого нужно сохранить исходные индексы, чтобы на этапе обратного распространения мы смогли
        поместить каждый градиент в нужное место, воспользовавшись простым циклом for.
        """
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)
        return softmax_output

    def cross_entropy(self, target_indices):
        """
        Вычисление softmax и потерь(loss) производится в одной функции.
        Намного быстрее вместе вычислить градиент softmax и отрицательный логарифм подобия в функции перекрестной
        энтропии, чем распространять их вперед и назад по отдельности в двух разных модулях.
        """
        # Определение функции softmax.
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)

        # Преобразование матрицы в (1, N). В примере (4, 1) в (1, 4).
        t = target_indices.data.flatten()
        # .reshape(len(t), -1) - не обязательно.
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * target_dist).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)


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


class CrossEntropyLoss(Layer):  # TODO Describe this
    """
    Теперь вычисление softmax и потерь производится в одном классе
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)


class RNNCell(Layer):
    """
    Рекуррентная нейронная сеть имеет вектор состояния, который передается из предыдущей итерации обучения в следующую.
    В данном случае это переменная hidden, которая одновременно является входным параметром и выходным значением
    функции forward.
    Рекуррентная нейронная сеть имеет также несколько разных весовых матриц: одна отображает входные векторы в скрытые
    (обработка входных данных), другая отображает скрытые векторы в скрытые векторы
    (коррекция каждого скрытого вектора на основе предыдущего состояния)
    и третья, необязательная, отображает скрытый слой в выходной, генерируя прогноз на основе скрытых векторов.
    Наша реализация RNNCell включает все три матрицы.
    Слой self .w_ih отображает входной слой в скрытый, self .w_hh — скрытый слой в скрытый и
    self .w_ho — скрытый слой в выходной.
    Обратите внимание на размерности всех трех матриц.
    Оба размера — n_input матрицы self .w_ih и n_output матрицы self .w_ho — определяются размером словаря.
    Все остальные размеры определяются параметром n_hidden.
    Параметр activation определяет нелинейную функцию для применения к скрытым векторам в каждой итерации.
    """

    def __init__(self, n_inputs, n_hidden, n_output, activation="sigmoid"):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        if activation == "sigmoid":
            self.activation = Sigmoid()

        elif activation == "tanh":
            self.activation = Tanh()

        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):

        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)


class Words2Vectors:
    identity = np.eye(5)
    # print(f"Вместо слов, мы должны передать индексы слов.", "\n", identity, "\n")
    # print(f"Возвращает ту же матрицу, но заменит каждое число в исходной матрице соответствующей строкой. \n"
    # f"Так, двумерная матрица индексов превратится в трехмерную матрицу векторных представлений (строк). \n",
    # identity[np.array([[1, 2, 3, 4],
    # [2, 3, 4, 0],
    # [4, 3, 2, 1]])])


class LSTMCell(Layer):
    """
    Слой долгой краткосрочной памяти LSTM.
    LSTM создает следующее скрытое состояние, копируя предыдущее, а затем
    добавляет или удаляет информацию по мере необходимости. Для добавления
    и удаления информации LSTM использует механизмы, которые называют
    вентилями, или фильтрами (gate).
    """

    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.xf = Linear(n_inputs, n_hidden)
        self.xi = Linear(n_inputs, n_hidden)
        self.xo = Linear(n_inputs, n_hidden)
        self.xc = Linear(n_inputs, n_hidden)

        self.hf = Linear(n_hidden, n_hidden)
        self.hi = Linear(n_hidden, n_hidden)
        self.ho = Linear(n_hidden, n_hidden)
        self.hcell = Linear(n_hidden, n_hidden)

        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()

        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()
        self.parameters += self.ho.get_parameters()
        self.parameters += self.hcell.get_parameters()

        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        """
        Логика прямого распространения в ячейке LSTM:
        Ячейка LSTM имеет два вектора со скрытым состоянием: h (от англ. hidden — скрытый) и cell.
        Каждое новое значение является суммой предыдущего значения и приращения 'и', взвешенных весами 'f' и 'i'.
        Здесь f — это «забывающий» («forget») вентиль (фильтр).
        Если этот вес получит значение 0, новая ячейка «забудет» то, что видела прежде.
        Если i получит значение 1, приращение 'и' будет полностью добавлено в новую ячейку.
        Переменная 'о' — это выходной вентиль (фильтр), который определяет,
        какая доля состояния ячейки попадет в прогноз.
        Например, если все значения в о равны нулю, тогда строка self.w_ho.forward(h)
        вернет прогноз, полностью игнорируя состояние ячейки.

        Они действуют вместе и гарантируют, что для корректировки информации, хранящейся в cell,
        не потребуется применять матричное умножение или нелинейную функцию активации.
        Иначе говоря, избавляют от необходимости вызывать nonlinearity(cell) или cell.dot(weights).

        Такой подход позволяет модели LSTM сохранять информацию на протяжении
        временной последовательности, не беспокоясь о затухании или взрывном росте
        градиентов. Каждый шаг заключается в копировании (если f имеет ненулевое
        значение) и прибавлении приращения (если i имеет ненулевое значение).
        Скрытое значение h — это замаскированная версия ячейки, используемая для получения прогноза.
        Обратите также внимание, что все три вентиля формируются совершенно одинаково.
        Они имеют свои весовые матрицы, но каждый зависит от входных
        значений и скрытого состояния, пропущенных через функцию sigmoid.
        Именно эта нелинейная функция sigmoid делает их полезными в качестве вентилей,
        преобразуя в диапазон от 0 до 1.

        И последнее критическое замечание в отношении вектора h. Очевидно, что он
        все еще подвержен эффекту затухания и взрывного роста градиентов, потому
        что фактически используется так же, как в простой рекуррентной нейронной
        сети. Во-первых, поскольку вектор h всегда создается из комбинации векторов,
        которые сжимаются с помощью tanh и sigmoid, эффект взрывного роста градиентов
        на самом деле отсутствует — проявляется только эффект затухания. Но
        в итоге в этом нет ничего страшного, потому что h зависит от ячейки cell, которая
        может переносить информацию на дальние расстояния: ту информацию, которую
        затухающие градиенты не способны переносить. То есть вся перспективная
        информация транспортируется с помощью cell, a h — это всего лишь локальная
        интерпретация cell, удобная для получения прогноза на выходе и активации
        вентилей на следующем шаге. Проще говоря, способность с переносить информацию
        на большие расстояния нивелирует неспособность h к тому же самому.
        """
        prev_hidden = hidden[0]
        prev_cell = hidden[1]

        # 3 вентиля f,i,o и вектор приращений u:
        # Механизм управления забыванием (forget).
        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        # Механизм управления вводом (input).
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        # Механизм управления выводом (output).
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
        # Механизм управления изменением (update).
        u = (self.xc.forward(input) + self.hcell.forward(prev_hidden)).tanh()

        cell = (f * prev_cell) + (i * u)
        h = o * cell.tanh()
        output = self.w_ho.forward(h)

        return output, (h, cell)

    def init_hidden(self, batch_size=1):
        init_h = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_cell = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_h.data[:, 0] += 1
        init_cell.data[:, 0] += 1
        return init_h, init_cell
