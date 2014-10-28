# coding=utf-8

__author__ = 'sejros'


class _Node(object):
    def __init__(self):
        self.parents = []
        self.conditional = None
        self.uncond = None
        self.name = ''

    def __repr__(self):
        return self.name


class Net(object):
    """ Данный класс реализует смешанную сеть вывода.

    Основным назначением данного класса является выполнение запросов к сети.
    Сеть формируется как набор факторов, характеризующих совместное распределение значений некоторого набора переменных.

    Сети Байса традиционно представляются в виде графа, в котором вершины представляют переменные, входящие в сеть,
    а ребра – причинно-следственные связи, причем ребро направлено от причины к следствию. Это очень наглядное
    представление является одним из главных достоинств вероятностных графических моделей и позволяет отобразить
    условную вероятность в виде взаимосвязей переменных и факторов, а также зачастую построить граф по экспертным или
    эмпирическим данным для моделирования распределения вероятностей. В графе наглядно видна иерархичность условной
    вероятности. Если некая переменная X зависит от переменной Y, то переменная Y будет среди родителей переменной X
    на графе.

    Запрос к байесовской сети выглядит следующим образом: каково распределение полной вероятности набора переменных Q,
    при условии, что набор переменных E принимает назначение q? Множество переменных Q называется запросом (query) или
    целевыми переменными и может состоять из одной и более переменных. Множество условий E называется
    наблюдения (evidence) или наблюдаемыми переменными и, в общем случае, может быть пустым. Множества Q и E не должны
    пересекаться. Множество переменных, входящих в байесовскую сеть, но не входящих во множества Q и E называется
    скрытые (hidden) переменные. Семантика этих множеств довольно очевидна. Запрос – это целевые переменные, которые
    нас интересуют, исходя из контекста конкретной задачи. Наблюдение – это те переменные, значения которых мы можем
    измерить или предсказать. Скрытые переменные не являются ни тем, ни другим, но могут оказывать неявное влияние на
    запрос и/или на наблюдения. В таких обозначениях использование сети Байеса для логического вывода сводится к
    вычислению вероятности P(Q|E).

    Достоинством сетей Байеса является универсальность. Единожды сконструированная, сеть может использоваться для
    вычисления любых корректных запросов на области ее определения, то есть не нужно изменять конструкцию сети,
    чтобы выполнять запросы определенного вида. Запрос является корректным, если выполняются два условия:
    - все переменные входящие в множества наблюдений и запросов входят в область определения сети;
    - множества Q и E не пересекаются.

    Итак, каждый запрос разбивает множество переменных области определения сети на три непересекающихся множества:
    Q, E и H. Значение любого запроса к Байсовской сети на этих множествах может быть вычислен только из фактора,
    представляющего распределение полной вероятности P(Q,E,H).

    Синтаксис:
        >>> import numpy as np
        >>> from pyinference.inference.variable import Variable
        >>> from pyinference.inference.factor import Factor
        >>> c = Variable(name='C', terms=['no', 'yes'])
        >>> t = Variable(name='T', terms=['pos', 'neg'])
        >>> c_node = Factor(name='C', cons=[c])
        >>> c_node.cpd = np.array([0.99, 0.01])
        >>> t_node = Factor(name='T|C', cons=[t], cond=[c])
        >>> t_node.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])
        >>> bn = Net(name='Cancer', nodes=[c_node, t_node])

    Поля класса:
        name (`str`): имя сети;

        nodes(`list`): список факторов, составляющих сеть.

    Именованные параметры:
        name (`str`): имя сети;

        nodes (`list`): список факторов, составляющих сеть.
            Передача конструктору списока ``bn = Node(name='sample net', nodes=a_list)`` эквивалентна использованию
            метода :func:`add_node`::

                bn = Net(name='sample net')
                for node in a_list:
                    bn.add_node(node)

            Поэтому при использовании конструктора может генерироваться исключение метода :func:`add_node`.
            В частности, такое может произойти при неверном порядке факторов в передаваемом списке. Поэтому,
            рекомендуется использовать конструктор без второго параметра, а факторы в сеть добавлять явно.
    """

    def __init__(self, name='', nodes=None):
        self.name = name
        self.nodes = []
        for node in (nodes or []):
            self.add_node(node)

    def joint(self):
        """ Рассчитывает распределение совместной вероятности всех переменных сети.

        Синтаксис:
            >>> import numpy as np
            >>> from pyinference.inference.variable import Variable
            >>> from pyinference.inference.factor import Factor
            >>> c = Variable(name='C', terms=['no', 'yes'])
            >>> t = Variable(name='T', terms=['pos', 'neg'])
            >>> c_node = Factor(name='C', cons=[c])
            >>> c_node.cpd = np.array([0.99, 0.01])
            >>> t_node = Factor(name='T|C', cons=[t], cond=[c])
            >>> t_node.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])
            >>> bn = Net(name='Cancer', nodes=[c_node, t_node])

            >>> j = bn.joint()
            >>> j.shape
            (2, 2)
            >>> len(j.vars)
            2
            >>> "%0.3f" % j.cpd[0,0]
            '0.198'
            >>> "%0.3f" % j.cpd[1,1]
            '0.001'

        Возвращает:
            Фактор (:class:`Factor`), представляющий рапределение полной вероятности всех
            переменных сети.
        """
        res = None
        for node in self.nodes:
            res *= node.conditional
        return res

    def add_node(self, factor):
        """ Метод добавляет фактор к сети.

        В процессе добавления фактора к сети производится проверка корректности. Он заключается в том, что мы не можем
        добаить к сети фактор, если его родителя в сети нет. Например, у нас есть два фактора: F(C) и G(T|C).
        Так как второй фактор (G) представляет условную вероятность, его родителем должен быть фактор, определяющий
        распределение вероятности переменной C, то есть, фактор F. Таким образом, если мы *сначала* попытаемся добавить
        к сети фактор G, то получим ошибку, так как его родителя в сети нет. Однако, если сперва добавить фактор
        F, а уже *затем* фактор G, то проблем не возникнет.
        Такая проверка гарантирует корректность графа, представляющего данную сеть.

        Синтаксис:
            >>> import numpy as np
            >>> from pyinference.inference.variable import Variable
            >>> from pyinference.inference.factor import Factor
            >>> c = Variable(name='C', terms=['no', 'yes'])
            >>> t = Variable(name='T', terms=['pos', 'neg'])
            >>> c_node = Factor(name='C', cons=[c])
            >>> c_node.cpd = np.array([0.99, 0.01])
            >>> t_node = Factor(name='T|C', cons=[t], cond=[c])
            >>> t_node.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])
            >>> bn = Net(name='Cancer')
            >>> bn.add_node(c_node)
            >>> bn.add_node(t_node)

        Параметры:
            factor (:class:`Factor`): добавляемый фактор

        Исключения:
            `AttributeError`: ошибка возникает, если при добавлении фактора провалилась проверка корректности.
        """
        correct = True
        node = _Node()
        node.conditional = factor
        node.name = factor.name
        for var in factor.cond:  # проверка, есть ли распределение этого фактора в сети
            found = False
            for node_ in self.nodes:
                for cons in node_.conditional.cons:
                    if var.name == cons.name:
                        found = True
                        node.parents.append(node_)
            correct = correct and found
        if not correct:
            raise AttributeError
        uncond = factor
        for parent in node.parents:
            uncond = uncond * parent.uncond
            uncond = uncond - parent.uncond.cons
        node.uncond = uncond
        self.nodes.append(node)

    def query(self, query=None, evidence=None):
        """ Выполняет запрос к сети вывода.

        Синтаксис:
            >>> import numpy as np
            >>> from pyinference.inference.variable import Variable
            >>> from pyinference.inference.factor import Factor
            >>> c = Variable(name='C', terms=['no', 'yes'])
            >>> t = Variable(name='T', terms=['pos', 'neg'])
            >>> c_node = Factor(name='C', cons=[c])
            >>> c_node.cpd = np.array([0.99, 0.01])
            >>> t_node = Factor(name='T|C', cons=[t], cond=[c])
            >>> t_node.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])
            >>> bn = Net(name='Cancer', nodes=[c_node, t_node])

            >>> q = bn.query(query=[c], evidence=[t])
            >>> len(q.vars)
            2
            >>> q.cond[0].name
            'T'
            >>> q.cons[0].name
            'C'
            >>> "%0.3f" % q.cpd[0,0]
            '0.957'
            >>> "%0.3f" % q.cpd[1,1]
            '0.001'

        Именованные параметры:
            query (`list`): список переменных (:class:`Variable`) запроса;

            evidence (`list`): список переменных (:class:`Variable`) свидетельств.

        Возвращает:
            Фактор (:class:`Factor`), представляющий рапределение условной вероятности,
            где условными переменными являются наблюдения (evidence), а подусловными - переменные запроса (query):
            F(Q|E).
        """
        query = query or []
        evidence = evidence or []
        # TODO локальный вывод
        res = self.joint()
        # TODO проверка корректности
        hidden = list(set(res.vars) - set(query) - set(evidence))
        for h in hidden:
            res -= h
        for e in evidence:
            for node in self.nodes:
                if node.uncond.vars == [e]:
                    res /= node.uncond
        return res