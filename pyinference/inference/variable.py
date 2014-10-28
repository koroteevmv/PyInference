# coding=utf-8

import numpy as np
from pyinference.fuzzy import set as fuzzy_set

__author__ = 'sejros'


class Variable(object):
    """ Класс реализует переменную

    Переменная представляет собой некий параметр, могущий иметь определенный набор значений.
    Набор значений переменной может задаваться двумя способами.
    Во-первых, можно задать списком. В таком случае, этот список следует передать как аргумент terms конструктора.
    Во-вторых, с переменной может быть ассоциирован нечеткий классификатор (см.
    :class:`pyinference.fuzzy.set.FuzzySet`).
    Тогда передать конструктору следует его.

    Переменная в байесовском моделировании – некая сущность, обладающая именем и областью определения.
    Обычно, рассматриваются переменные двух типов: дискретные и непрерывные. Дискретные переменные принимают значения
    из некоторого конечного множества X, а непрерывные – определены на некотором подмножестве множества действительных
    чисел. В общем случае, переменная определяется упорядоченной парой V=(N, X), где N – имя переменной, а X –
    множество возможных значений.

    Примером случайной переменной может быть результат подбрасывания монеты в таком случае областью ее определения
    будет множество {“орел”, “решка”}. В общем случае, конкретные значения переменной не имеют синтаксического значения
    (имеют место только семантически, то есть по смыслу) и их обычно заменяют соответствующим по числу элементов
    подмножество множества натуральных числе. То есть в нашем примере, можно обозначить значение «решка» за 0, а
    «орел» - за 1, и тогда область определения переменной будет {0, 1}.

    Построим данную переменную с использованием класса Variable::

        coin = Variable(name="coin", terms=["орел", "решка"])

    Частным и довольно распространенным случаем дискретной переменной является переменная, способная принимать только
    два значения. Такая переменная называется бинарной. В примере выше – результат подбрасывания монеты – бинарная
    переменная.

    Наиболее важным производным свойством дискретной переменной является мощность переменной – количество значений,
    которые она может принимать. Мощность бинарной переменной равна двум.

    Синтаксис:
        >>> import pyinference.inference.variable
        >>> a = Variable(name='a', terms=[0, 1])
        >>> a.value = 'low'

        >>> from pyinference.fuzzy import set as fuzzy_set
        >>> fs = fuzzy_set.Partition(peaks=[0.0, 0.5, 1.0])
        >>> b = Variable(name='B', terms=fs)

    Поля класса:
        card (`int`): мощность переменной (количество значений)

        classifier(`dict` or :class:`pyinference.fuzzy.set.FuzzySet`): связанный с переменной классификатор

        name (`str`): имя переменной

        terms (`list`): список значений переменной (терм-множество)

        value (`object`): текущее значение переемнной

    Именованные параметры:
        name (`str`): имя переменной

        terms (`list` or `dict` or :class:`pyinference.fuzzy.set.FuzzySet`): набор значений переменной

    Исключения:
        `TypeError`: ошибка возникает, если агрумент terms имеет неподдерживаемый тип.
    """

    def __init__(self, name='', terms=None):
        self.terms = []
        self.classifier = {}
        if isinstance(terms, list):
            self.terms = terms
            self.classifier = {}
        elif isinstance(terms, fuzzy_set.FuzzySet):
            self.terms = terms.sets
            self.classifier = terms
        elif isinstance(terms, dict):
            self.terms = terms.keys()
            self.classifier = terms
        else:
            raise TypeError
        self.card = len(self.terms)
        self.value = None
        self.name = name

    def equals(self, value):
        """Проверка переменной на равенство значению.

        Данный метод принимает некое значение и вычисляет меру сходства его со значением атрибута `value`.

        Синтаксис:
            >>> import pyinference.inference.variable
            >>> a = Variable(name='a', terms=[0, 1])
            >>> a.value = 'low'
            >>> '%.2f' % a.equals('low')
            '1.00'
            >>> '%.2f' % a.equals('high')
            '0.00'

            >>> from pyinference.fuzzy import set as fuzzy_set
            >>> import pyinference.inference.variable
            >>> fs = fuzzy_set.Partition(peaks=[0.0, 0.5, 1.0])
            >>> b = Variable(name='B', terms=fs)
            >>> b.value = 0.5
            >>> '%.2f' % b.equals(0.5)
            '1.00'
            >>> b.value = '1'
            >>> '%.2f' % b.equals(0.25)
            '0.50'

        Параметры:
            value (`object`): Значение переменной.
            Может быть как элементом терм-множества занчений, явно перечисленного в атрибуте `terms` класса,
            так и элементом области определения связанного с этой переменной классификатора.

        Возвращает:
            Действительное число x, где -1.0 <= x <= 1.0, характеризующую меру сходства переданного и текущего
            значений переменной.

        Исключения:
            `AttributeError`: если текущее значение переменной не было задано до вызова данного метода.
        """
        if self.value is None:
            raise AttributeError
        if self.value in self.classifier:
            val1 = self.classifier[self.value]
        else:
            val1 = self.value
        if value in self.classifier:
            val2 = self.classifier[value]
        else:
            val2 = value
        return float(val1 == val2)

    def __repr__(self):
        """ Краткое текстовое представление перееменной.

        Синтаксис:
            >>> import pyinference.inference.variable
            >>> a = Variable(name='A', terms=[0, 1])
            >>> print a
            A

        Возвращает:
            Возвращает имя переменной
        """
        return self.name
