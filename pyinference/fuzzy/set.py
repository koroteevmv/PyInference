# -*- coding: UTF-8 -*-

""" Модуль для работы с нечеткими множествами.

Модуль реализует функциональность аппарата нечеткой логики в части работы с
нечеткими множествами. Он включает:

    - абстрактный класс нечеткого множества,
    - шаблоны для создания классификаторов различных видов.

В основном, модуль предназначен длясоздания и спользования нечетких
классификаторов. С помощью него можно создать классификатор как в ручном режиме
и заполнить его термами самостоятельно, так и воспользоваться одним из
конструкторов, описанных ниже.

Нечеткий классификатор - это одно из применений нечеткого множества. Он состоит
из нескольких нечетких подмножеств (см. :class:`pyinference.fuzzy.subset.FuzzySubset`), определенных на одном
носителе (интервале определения). Термы множества имеют метки,которые
используются в качестве значений лингвистических переменных вместо обычных
чисел.
"""

from pyinference.fuzzy.subset import Trapezoidal
from pyinference.fuzzy.subset import Gaussian
from pyinference.fuzzy.subset import Triangle
from pyinference.fuzzy.subset import Subset
import pyinference.fuzzy.domain

import math
import pylab as p


class FuzzySet(object):
    """Нечеткое множество.

    Нечеткое множество, или классификатор, состоящее из набора нечетких
    подмножеств, определенных на одном и том же носителе.

    Синтаксис:
        >>> A = FuzzySet(0, 100, name='Classifier')
        >>> A.domain.begin
        0.0
        >>> A.domain.end
        100.0
        >>> A.name
        'Classifier'
        >>> A.sets
        {}

    Поля класса:
        sets (`dict`): Ассоциативный массив, содержащий, соответственно, имя и объект типа
            :class:`pyinference.fuzzy.subset.Subset`, для каждого терма нечеткого множества.

        domain (:class:`pyinference.fuzzy.domain.Domain`): носитель нечеткого множества

        name (`str`): имя классификатора

    Именованные параметры:
        begin (`float`): начало интервала определения классификатора

        end (`float`): конец интервала определения классификатора

        name (`str`): имя классификатора

        domain (:class:`pyinference.fuzzy.domain.Domain`): носитель нечеткого множества
    """

    def __init__(self, begin=0.0, end=1.0, domain=None, name=''):
        if not domain:
            domain = pyinference.fuzzy.domain.RationalRange(begin, end)
        self.domain = domain
        self.sets = {}
        self.name = name

    def __iter__(self):
        """ Процедура перебора термов классификатора.

        Syntax:
            >>> C = Partition(peaks=[0.0, 0.3, 1.0])
            >>> for i in C:
            ...     print '%0.3f' % i.centr()
            ...
            0.433
            0.100
            0.766
        """
        for i in self.sets.iterkeys():
            yield self[i]

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, param):
        """
        Для быстрого доступа к подмножеству нечеткого множества, терму
        классификатора или значению лингвистической переменной можно
        использовать следующий синтаксис:

        Syntax:
            >>> C  =  Partition(peaks=[0.0, 0.3, 1.0])
            >>> '%0.3f' % C['0'].centr()
            '0.100'
            >>> '%0.3f' % C['0'].card()
            '0.150'
            >>> C['0'].domain.begin
            0.0
            >>> C['1'].mode()
            0.3
            >>> C['2'].mode()
            1.0
            >>> C['0'].mode()
            0.0
        """
        return self.sets[param]

    def __setitem__(self):
        raise NotImplementedError

    def __delitem__(self):
        raise NotImplementedError

    def __contains__(self, value):
        return value in self.sets

    def keys(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def has_key(self):
        raise NotImplementedError

    def iterkeys(self):
        raise NotImplementedError

    def itervalues(self):
        raise NotImplementedError

    def iteritems(self):
        raise NotImplementedError

    def add_term(self, sub, name=''):
        """ Добавляет терм к данному классификатору. Порядок термов не важен.

        Syntax:
            >>> A = FuzzySet(0, 100)
            >>> S = Gaussian(20, 10)
            >>> A.add_term(S, name = 'term1')
            >>> A.sets['term1'].median
            20.0
            >>> A.add_term(Triangle(30, 50, 75), name = 'term2')
            >>> A.sets['term2'].mode()
            50.0

        Параметры:
            sub: нечеткое подмножество типа Subset, или любого производного,
                играющее роль терма нечеткого множества (классификатора)

        Именованные параметры:
            name: cтрока, идентифицирующая терм в составе данного множества.
                Используется для построении легенды в методе plot(), а также как
                ключ ассоциативного массива Sets

        """
        self.sets[name] = sub

    def find(self, val, term):
        """Возвращает значение принадлежности точки x терму term.

        Синтаксис:
            >>> C  =  Partition(peaks=[0.0, 0.3, 1.0])
            >>> C.find(0, '0')
            1.0
            >>> '%0.3f' % C.find(0.12, '0')
            '0.600'
            >>> C.find(0.12, '1')
            0.4
            >>> C.find(0.12, '2')
            0.0
            >>> C.find(0.5, '0')
            0.0
            >>> round(C.find(0.65, '1'), 3)
            0.5
            >>> round(C.find(0.65, '2'), 3)
            0.5

        Параметры:
            val (`float` or `object`): элемент области определения нечеткого множества

            term (`str`): имя терма;

        Возвращает:
            Число, соответствующее принадлежности данной точке области определения (val) данному терму
            классифиактора (term).

        """
        try:
            return self.sets[term].value(val)
        except KeyError:
            return None

    def classify(self, val):
        """ Возвращает имя терма, наиболее соответствующего переданному элементу.

        Будучи вызванным у квалификатора, соответствует квалификации точного
        значения или значения, выраженного нечетким подмножеством или числом.

        Синтаксис:
            >>> from pyinference.fuzzy.subset import Gaussian, Triangle
            >>> A = FuzzySet(0, 100, name='Classifier')
            >>> A.add_term(Gaussian(20, 10), name='term1')
            >>> A.add_term(Triangle(30, 50, 75), name='term2')
            >>> A.classify(15)
            'term1'
            >>> A.classify(55)
            'term2'
            >>> A.classify(40)
            'term2'

        Параметры:
            val (`float` or `object`): элемент области определения нечеткого множества.

        Возвращает:
            Имя классификатора, наиболее соответствующего данной точке.

        """
        res = {}
        if isinstance(val, Subset):
            for i in self.sets.iterkeys():
                j = val & self.sets[i]
                res[i] = j.card()
        else:
            for i in self.sets.iterkeys():
                j = self.sets[i].value(val)
                res[i] = j
        maxim = 0
        name = None
        for i in res.iterkeys():
            if res[i] > maxim:
                maxim = res[i]
                name = i
        return name

    def plot(self, verbose=False, subplot=p):
        """ Отображает нечеткое множество графически. Все термы представляются на одном графике.

        Синтаксис:
            >>> C  =  Partition(peaks=[0.0, 0.3, 1.0])
            >>> C.plot()

        Именованные параметры:
            verbose (`bool`): задает подробное указание критических точек подмножеств на графике;

            subplot (:class:`pylab.plot`): если требуется отобразить данное множество на подграфике,
                можно воспользоваться данным параметром.
        """
        labels = []
        for name, sub in self.sets.iteritems():
            sub.plot(verbose=verbose, subplot=subplot)
            labels.append(name)
        subplot.legend(labels, loc='upper right')
        subplot.plot(self.domain.begin, 1.01)
        subplot.plot(self.domain.end + (self.domain.end - self.domain.begin) / 5, -0.01)
        subplot.grid()


class TriangleClassifier(FuzzySet):
    """Равномерный классификатор с термами в виде равноскатных треугольных чисел.

    Синтаксис:
        >>> A = TriangleClassifier(names=['low', 'middle', 'high'])


        >>> A  =  TriangleClassifier(begin = 0, end = 100,
        ...                             name = 'Sample classifier',
        ...                             names = ['low', 'middle', 'high'],
        ...                             edge = True, cross = 2)
        >>> print A.domain.begin
        0.0
        >>> print A.domain.end
        100.0
        >>> print A['low']
        25.0
        >>> print A['low'].mode()
        25.0
        >>> print A['middle'].mode()
        50.0
        >>> print A['high'].mode()
        75.0
        >>> A  =  TriangleClassifier(names = ['low', 'middle', 'high'])
        >>> print A.domain.begin
        0.0
        >>> print A.domain.end
        1.0
        >>> print A['low'].mode()
        0.0
        >>> print A['middle'].mode()
        0.5
        >>> print A['high'].mode()
        1.0

    Параметры конструктора:
        names (`list`): список строк, каждая из которых представляет собой имя терма,
             входящего в данный классификатор. Порядок термов соблюдается.

        edge (`bool`): параметр, задающий расположение термов.
             Если edge = False (**по умолчанию**), то первый и последний термы будут иметь вершины
             в точках, соответственно, начала и конца интервала определения
             классификатора:

             >>> names = ['1', '2', '3']
             >>> A = TriangleClassifier(names = names, edge = False)
             >>> A.sets['1'].mode()
             0.0
             >>> A.sets['2'].mode()
             0.5
             >>> A.sets['3'].mode()
             1.0

             Если edge = True, то первый и последний термы будут иметь отступ
             от границ интервала определения, равный отступу между термами:

             >>> A = TriangleClassifier(names = names, edge = True)
             >>> '%0.3f' % A.sets['1'].mode()
             '0.167'
             >>> '%0.3f' % A.sets['2'].mode()
             '0.500'
             >>> '%0.3f' % A.sets['3'].mode()
             '0.833'

        cross: Параметр задает степень пересечения термов классификатора.
             Может принимать значения от 0 до бесконечности.
             При cross = 1 (**по умолчанию**) каждый объект области определения принадлежит только
             одному множеству.
             Функции принадлежности соседних термов расположены "впритык".

             >>> A = TriangleClassifier(names = names)
             >>> A.sets['1'].domain.end
             0.25
             >>> A.sets['2'].domain.begin
             0.25
             >>> A.sets['2'].domain.end
             0.75
             >>> A.sets['3'].domain.begin
             0.75

             При cross = 2 функции принадлежности строятся таким образом, что каждый
             терм покрывает ровно половину ширины соседних термов. Таким образом,
             каждая точка области определения принадлежит двум нечетким
             подмножествам.

             >>> A = TriangleClassifier(names = names, cross = 2.0)
             >>> A.sets['1'].domain.end
             0.5
             >>> A.sets['2'].mode()
             0.5
             >>> A.sets['2'].domain.end
             1.0
             >>> A.sets['3'].domain.begin
             0.5
             >>> A.sets['3'].mode()
             1.0

             При 0 < cross < 1 между термами классификатора появляются интервалы,
             значения в которых не принадлежат ни одному терму.
             При cross = 0 термы вырождаются в точку.
    """

    def __init__(self, begin=0.0,
                 end=1.0,
                 domain=None,
                 name='',
                 names=None,
                 edge=False,
                 cross=1.0):
        if not domain:
            domain = pyinference.fuzzy.domain.RationalRange(begin, end)
        super(TriangleClassifier, self).__init__(domain=domain, name=name)

        if not names:
            raise ValueError
        if not edge:
            # edge = False
            wide = (end - begin) * cross / (len(names) - 1)
            step = (end - begin) / (len(names) - 1)
            wide /= 2
            mode = 0
        else:
            # edge = True
            wide = (end - begin) / (len(names) + 1 - cross)
            wide /= 2
            mode = wide
            step = 2 * wide / cross
        for name in names:
            self.add_term(Triangle(mode - wide, mode, mode + wide), name=name)
            mode += step


class GaussianClassifier(FuzzySet):
    """Равномерный классификатор с термами в виде гауссиан.

    Синтаксис:
        >>> A = GaussianClassifier(names=['low', 'middle', 'high'])

        >>> A  =  GaussianClassifier(names = ['low', 'middle', 'high'])
        >>> print A.domain.begin
        0.0
        >>> print A.domain.end
        1.0
        >>> print A['low'].mode()
        0.0
        >>> print A['middle'].mode()
        0.5
        >>> print A['high'].mode()
        1.0

    Параметры конструктора (см. :class:`pyinference.fuzzy.set.FuzzySet`,
    :class:`pyinference.fuzzy.set.TriangleClassifier`).

    ..note::
        Так как гауссиана не ограничена по области определения, параметр `cross` конструктора данного класса
        ограничивает область в три сигмы.

    """

    def __init__(self, begin=0.0,
                 end=1.0,
                 domain=None,
                 name='',
                 names=None,
                 edge=0,
                 cross=1.0):
        if not domain:
            domain = pyinference.fuzzy.domain.RationalRange(begin, end)
        super(GaussianClassifier, self).__init__(domain=domain, name=name)

        if not names:
            raise ValueError
        if not edge:
            # edge = False
            wide = (end - begin) * cross / (len(names) - 1)
            step = (end - begin) / (len(names) - 1)
            wide /= 2
            mode = 0
        else:
            # edge = True
            wide = (end - begin) / (len(names) + 1 - cross)
            wide /= 2
            mode = wide
            step = 2 * wide / cross
        for name in names:
            self.add_term(Gaussian(mode, wide / 3), name=name)
            mode += step


class Partition(FuzzySet):
    """ Данный класс создает линейный неравномерный классификатор по точкам, указанным в параметрах.

    Характерной особенностью данного классификатора
    являтеся то, что для каждого элемента носителя сумма принадлежностей всех
    термов равна 1,0. Классификатор строится на действительном интервале.
    Термы классификатора именуются арабскими числами, начиная с 1.

    Синтаксис:
        >>> Clas = Partition(peaks=[0.0, 0.1, 0.3, 0.4, 0.6, 1.0], overlap=0.2)

        >>> A = Partition(begin=10, end=20, peaks=[10, 13, 18, 20], overlap=1.0, name='sample classifier')
        >>> A['1'].mom()
        13.0
        >>> A = Partition(begin=10, end=20, peaks=[10, 13, 18, 20], overlap=0.2, name='sample classifier')
        >>> '%0.3f' % A['1'].mom()
        '13.430'
        >>> A.classify(14)
        '1'
        >>> A.classify(17)
        '2'

        >>> A  =  Partition(begin = 10, end = 20, peaks = [10, 13, 15, 20], overlap = 0.2, name = 'sample classifier')
        >>> A.domain.begin
        10.0
        >>> A.domain.end
        20.0
        >>> A.name
        'sample classifier'
        >>> A.sets.keys()
        ['1', '0', '3', '2']
        >>> A['0'].mode()
        10.0
        >>> A['0'].domain.begin
        10.0
        >>> '%0.3f' % A['0'].domain.end
        '11.710'
        >>> '%0.3f' % A['1'].mode()
        '11.710'
        >>> '%0.3f' % A['1'].domain.begin
        '11.290'
        >>> '%0.3f' % A['1'].domain.end
        '14.140'

    Параметры:
        name (`str`): имя классификатора

        begin (`float`): начало области определения

        end (`float`): конец области определения

        domain (:class:`pyinference.fuzzy.domain.Domain`): носитель нечеткого множества

        peaks (`list`): массив чисел, представляющих центры интервалов толерантности термов

        overlap (`float`): параметр, задающий крутизну скатов ФП термов и ширину интервала
            толерантности. При overlap = 0 классификатор становится четким, при
            overlap = 1 ФП термов становятся треугольными.
    """

    def __init__(self, begin=0.0, end=1.0, domain=None,
                 peaks=None, overlap=1.0, name=''):
        if not peaks:
            peaks = []

        if not domain:
            domain = pyinference.fuzzy.domain.RationalRange(begin, end)
        super(Partition, self).__init__(domain=domain, name=name)

        peaks = sorted(peaks)
        peaks.insert(0, begin)
        peaks.append(end)
        overlap = math.tan(float(overlap) * math.pi / 2)
        for i in range(len(peaks) - 2):
            left = (peaks[i + 1] - peaks[i]) / (overlap + 2)
            right = (peaks[i + 2] - peaks[i + 1]) / (overlap + 2)
            begin = peaks[i + 1] - left * (1 + overlap)
            begin_tol = peaks[i + 1] - left
            end_tol = peaks[i + 1] + right
            end = peaks[i + 1] + right * (1 + overlap)
            self.add_term(Trapezoidal((begin, begin_tol, end_tol, end)),
                          name=str(i))

