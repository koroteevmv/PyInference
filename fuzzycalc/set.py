# -*- coding: UTF-8 -*-

''' Модуль для работы с нечеткими множествами

Модуль реализует функциональность аппарата нечеткой логики в части работы с
нечеткими множествами. Он включает:
    - абстрактный класс нечеткого множества,
    - шаблоны для создания классификаторов различных видов.

В основном, модуль предназначен длясоздания и спользования нечетких
классификаторов. С помощью него можно создать классификатор как в ручном режиме
и заполнить его термами самостоятельно, так и воспользоваться одним из
конструкторов, описанных ниже.

Нечеткий классификатор - это одно из применений нечеткого множества. Он состоит
из нескольких нечетких подмножеств (см. FuzzySubset), определенных на одном
носителе (интервале определения). Термы множества имеют метки,которые
используются в качестве значений лингвистических переменных вместо обычных
чисел.
'''

from .subset import Trapezoidal, Gaussian, Triangle, Subset
from .domain import RationalRange

import math
import pylab as p

class FuzzySet(object):
    '''
    Нечеткое множество, или классификатор, состоящее из набора нечетких
    подмножеств, определенных на одном и том же носителе.
    Синтаксис:
    >>> A = FuzzySet(0, 100, name = 'Classifier')
    >>> A.begin
    0.0
    >>> A.end
    100.0
    >>> A.name
    'Classifier'
    >>> A.Sets
    {}

    Параметры конструктора:
        begin
             начало интервала определения классификатора
        end
             конец интервала определения классификатора
        name
             имя классификатора
        domain

    Поля класса:
        Sets
            Ассоциативный массив, содержащий, соответственно, имя и объект типа
            Subset, для каждого терма нечеткого множества.
        domain
        name
    '''

    def __init__(self, begin=0.0, end=1.0, domain=None, name=''):
        '''
            >>> A  =  FuzzySet(0, 10, 'Sample fuzzy set')
            >>> print A.begin
            0.0
            >>> print A.end
            10.0
            >>> print A.name
            Sample fuzzy set

        '''
        if not domain:
            domain = RationalRange(begin, end)
        self.domain = domain
        self.sets = {}
        self.name = name

    def __iter__(self):
        '''Процедура перебора термов классификатора.
        Синтаксис:
            >>> C  =  Classifier(p = [0.0, 0.3, 1.0])
            >>> for i in C:
            ...     print i
            ...
            0.433333333333
            0.100066666667
            0.766822222222
            >>> for i in C:
            ...     print i.centr()
            ...
            0.433333333333
            0.100066666667
            0.766822222222

        '''
        for i in self.sets.iterkeys():
            yield self[i]

    def __getitem__(self, param):
        '''
        Для быстрого доступа к подмножеству нечеткого множества, терму
        классификатора или значению лингвистической переменной можно
        использовать следующий синтаксис:
            >>> C  =  Classifier(p = [0.0, 0.3, 1.0])
            >>> C['0'].centr()
            0.10006666666666648
            >>> C['0'].card()
            0.15000000000000002
            >>> C['0'].begin
            0.0
            >>> C['1'].mode()
            0.3
            >>> C['2'].mode()
            1.0
            >>> C['0'].mode()
            0.0


        '''
        return self.sets[param]

    def add_term(self, sub, name=''):
        '''
        Добавляет терм к данному классификатору. Порядок термов не важен.
        Синтаксис:
            >>> A = FuzzySet(0, 100)
            >>> S = Gaussian(20, 10)
            >>> A.add_term(S, name = 'term1')
            >>> A.Sets['term1'].mu
            20.0
            >>> A.add_term(Triangle(30, 50, 75), name = 'term2')
            >>> A.Sets['term2'].b
            50.0

        Параметры:
        S
            Нечеткое подмножество типа Subset, или любого производного,
            играющее роль терма нечеткого множества (классификатора)
        name
            Строка, идентифицирующая терм в составе данного множества.
            Используется для построении легенды в методе plot(), а также как
            ключ ассоциативного массива Sets

        '''
        self.sets[name] = sub

    def find(self, val, term):
        '''
        Возвращает значение принадлежности точки x терму term
        Синтаксис:
            >>> C  =  Classifier(p = [0.0, 0.3, 1.0])
            >>> C.find(0, '0')
            0.0
            >>> C.find(0.12, '0')
            0.6
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

        '''
        return self.sets[term].value(val)

    def classify(self, val):
        '''
        Возвращает имя терма, наиболее соответствующего переданному элементу.
        Будучи вызванным у квалификатора, соответствует квалификации точного
        значения или значения, выраженного нечетким подмножеством или числом.
        Синтаксис:
            >>> C = std_5_classificator()
            >>> C.classify(0.2)
            'II'
            >>> C.classify(0.8)
            'IV'
            >>> C.classify(1.0)
            'V'
            >>> C.classify(0.0)
            'I'
            >>> C.classify(Triangle(0.4, 0.5, 0.6))
            'III'
            >>> C.classify(Triangle(0.4, 0.5, 1.6))
            'IV'
            >>> C.classify(Triangle(-1.4, 0.5, 0.6))
            'II'
            >>> C.classify(Triangle(-1.4, 0.0, 0.6))
            'I'
        '''
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

    def plot(self):
        '''
        Отображает нечеткое множество графически. Все термы представляются на
        одном графике.
        Синтаксис:
            >>> C  =  Classifier(p = [0.0, 0.3, 1.0])
            >>> C.plot()
        '''
        labels = []
        for name, sub in self.sets.iteritems():
            sub.plot(verbose=False)
            labels.append(name)
        p.legend(labels, loc='upper right')
        p.plot(self.domain.begin, 1.01)
        p.plot(self.domain.end+(self.domain.end-self.domain.begin)/5, -0.01)
        p.title(self.name)
        p.grid()

class TriangleClassifier(FuzzySet):
    '''
    Равномерный классификатор с термами в виде равноскатных треугольных чисел.

    Синтаксис:
    >>> A = TriangleClassifier(begin = 0.0, end = 1.0,
                                name = '', names = [],
                                edge = 0, cross = 1)

    Параметры конструктора (см. FuzzySet):
    names
         список строк, каждая из которых представляет собой имя терма,
         входящего в данный классификатор. Порядок термов соблюдается.
    edge
         параметр, задающий расположение термов.
         Если edge = False, то первый и последний термы будут иметь вершины
         в точках, соответственно, начала и конца интервала определения
         классификатора:
         >>> names = ['1', '2', '3']
         >>> A = TriangleClassifier(names = names, edge = 0)
         >>> A.Sets['1'].b
         0.0
         >>> A.Sets['2'].b
         0.5
         >>> A.Sets['3'].b
         1.0

         Если edge = True, то первый и последний термы будут иметь отступ
         от границ интервала определения, равный отступу между термами:
         >>> A = TriangleClassifier(names = names, edge = 1)
         >>> A.Sets['1'].b
         0.25
         >>> A.Sets['2'].b
         0.5
         >>> A.Sets['3'].b
         0.75

    cross
         Параметр задает степень пересечения термов классификатора.
         Может принимать значения от 0 до бесконечности.
         При cross = 1 каждый объект области определения принадлежит только
         одному множеству.
         Функции принадлежности соседних термов расположены "впритык".
         >>> A = TriangleClassifier(names = names, cross = 1.0)
         >>> A.Sets['1'].c
         0.25
         >>> A.Sets['2'].a
         0.25
         >>> A.Sets['2'].c
         0.75
         >>> A.Sets['3'].a
         0.75

         При cross = 2 функции принадлежности строятся таким образом, что каждый
         терм покрывает ровно половину ширины соседних термов. Таким образом,
         каждая точка области определения принадлежит двум нечетким
         подмножествам.
         >>> A = TriangleClassifier(names = names, cross = 2.0)
         >>> A.Sets['1'].c
         0.5
         >>> A.Sets['2'].b
         0.5
         >>> A.Sets['2'].c
         1.0
         >>> A.Sets['3'].a
         0.5
         >>> A.Sets['3'].b
         1.0

         При 0 < cross < 1 между термами классификатора появляются интервалы,
         значения в которых не принадлежат ни одному терму.
         При cross = 0 термы вырождаются в точку.
    '''

    def __init__(self, begin=0.0,
                        end=1.0,
                        domain=None,
                        name='',
                        names=None,
                        edge=False,
                        cross=1.0):
        '''
        Синтаксис:
            >>> A  =  TriangleClassifier(begin = 0, end = 100,
                                            name = 'Sample classifier',
                                            names = ['low', 'middle', 'high'],
                                            edge = True, cross = 2)
            >>> print A.begin
            0.0
            >>> print A.end
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
            >>> print A.begin
            0.0
            >>> print A.end
            1.0
            >>> print A['low'].mode()
            0.0
            >>> print A['middle'].mode()
            0.5
            >>> print A['high'].mode()
            1.0

        '''
        if not domain:
            domain = RationalRange(begin, end)
        super(TriangleClassifier, self).__init__(domain=domain, name=name)

        if not names:
            raise ValueError
        if not edge:
            # edge = False
            wide = (end-begin) * cross/(len(names)-1)
            step = (end-begin)/(len(names)-1)
            wide = wide / 2
            mode = 0
        else:
            # edge = True
            wide = (end - begin) / (len(names) + 1 - cross)
            wide = wide / 2
            mode = wide
            step = 2 * wide / cross
        for name in names:
            self.add_term(Triangle(mode-wide, mode, mode+wide), name=name)
            mode = mode+step

class GaussianClassifier(FuzzySet):
    '''
    Равномерный классификатор с термами в виде гауссиан.

    Синтаксис:
    >>> A = GaussianClassifier(begin = 0.0, end = 1.0,
                                name = '', names = [],
                                edge = 0, cross = 1)

    Параметры конструктора (см. FuzzySet, TriangleClassifier):
    '''

    def __init__(self, begin=0.0,
                        end=1.0,
                        domain=None,
                        name='',
                        names=None,
                        edge=0,
                        cross=1.0):
        '''
        Синтаксис:
            >>> A  =  GaussianClassifier(names = ['low', 'middle', 'high'])
            >>> print A.begin
            0.0
            >>> print A.end
            1.0
            >>> print A['low'].mode()
            0.0
            >>> print A['middle'].mode()
            0.5
            >>> print A['high'].mode()
            1.0
        '''
        if not domain:
            domain = RationalRange(begin, end)
        super(GaussianClassifier, self).__init__(domain=domain, name=name)

        if not names:
            raise ValueError
        if not edge:
            # edge = False
            wide = (end-begin) * cross/(len(names)-1)
            step = (end-begin)/(len(names)-1)
            wide = wide / 2
            mode = 0
        else:
            # edge = True
            wide = (end - begin) / (len(names) + 1 - cross)
            wide = wide / 2
            mode = wide
            step = 2 * wide / cross
        for name in names:
            self.add_term(Gaussian(mode, wide/3), name=name)
            mode = mode+step


class Partition(FuzzySet):
    '''
    Данный класс создает линейный неравномерный классификатор по точкам,
    указанным в параметрах. Характерной особенностью данного классификатора
    являтеся то, что для каждого элемента носителя сумма принадлежностей всех
    термов равна 1,0. Классификатор строится на действительном интервале.
    Термы классификатора именуются арабскими числами, начиная с 1.
    Синтаксис:
    >>> Clas = Partition(p = [0.0, 0.1, 0.3, 0.4, 0.6, 1.0], u = 0.2)

    Параметры:
    name
        имя классификатора
    begin
        начало области определения
    end
        конец области определения
    p
        массив чисел, представляющих центры интервалов толерантности термов
    overlap
        параметр, задающий крутизну скатов ФП термов и ширину интервала
        толерантности. При overlap = 0 классификатор становится четким, при
        overlap = 1 ФП термов становятся треугольными.
    '''
    def __init__(self, begin=0.0, end=1.0, domain=None,
                        peaks=None, overlap=1.0, name=''):
        '''
        >>> A  =  Partition(begin = 10, end = 20,
                             peaks = [10, 13, 15, 20],
                             overlap = 0.2, name = 'sample classifier')
        >>> print A.begin
        10.0
        >>> print A.end
        20.0
        >>> print A.name
        sample classifier
        >>> print A.Sets.keys()
        ['1', '0', '3', '2']
        >>> print A['0'].mode()
        10.0
        >>> print A['0'].begin
        10.0
        >>> print A['0'].end
        11.709632851
        >>> print A['1'].mode()
        11.709632851
        >>> print A['1'].begin
        11.290367149
        >>> print A['1'].end
        14.139755234
        '''
        if not peaks:
            peaks = []

        if not domain:
            domain = RationalRange(begin, end)
        super(Partition, self).__init__(domain=domain, name=name)

        peaks = sorted(peaks)
        peaks.insert(0, begin)
        peaks.append(end)
        overlap = math.tan(float(overlap)*math.pi/2)
        for i in range(len(peaks)-2):
            left = (peaks[i+1]-peaks[i])/(overlap+2)
            right = (peaks[i+2]-peaks[i+1])/(overlap+2)
            begin = peaks[i+1]-left*(1+overlap)
            begin_tol = peaks[i+1]-left
            end_tol = peaks[i+1]+right
            end = peaks[i+1]+right*(1+overlap)
            self.add_term(Trapezoidal((begin, begin_tol, end_tol, end)),
                            name=str(i))

if __name__ == "__main__":
    import doctest
#    doctest.testmod(verbose=False)
    doctest.testmod(verbose=True)
