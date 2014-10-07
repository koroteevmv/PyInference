# -*- coding: UTF-8 -*-
''' Модуль для работы с нечеткими числами

Модуль реализует функциональность аппарата нечеткой логики в части работы с
нечеткими числами. Он включает:
    - описание
'''

from .subset import Subset
from math import tanh, log, pi, cos, exp, cosh

LINE = lambda c: lambda a, b: lambda x: abs((x-b)/(a-b))**c
LINES = lambda c: lambda a, b: lambda x: abs((x-b)/(a-b)) ** \
                        log(0.5, ((c-b)/(a-b)))
QUAD = lambda c: lambda a, b: lambda x: abs(1-((x-a)/abs(b-a))**2)**c
QUADS = lambda c: lambda a, b: lambda x: abs(1-((x-a)/abs(b-a))**2) ** \
                        log(0.5, (1-((c-a)/abs(b-a))**2))
LAPL = lambda c: lambda a, b: lambda x: exp(-abs(x-a)*c/abs(b-a))
LAPLS = lambda c: lambda a, b: lambda x: exp(abs(x-a)*log(0.5)/abs(c-a))
TANG = lambda c: lambda a, b: lambda x: (1+tanh(-((x-a)/(abs(b-a)/c))**2))
GAUS = lambda c: lambda a, b: lambda x: (exp(-((x-a)**2)/c**2))
CAUC = lambda c: lambda a, b: lambda x: (1/(1+((x-a)*c/abs(b-a))**2))
LOGI = lambda c: lambda a, b: lambda x: (2/(1+exp(((x-a)*c/abs(b-a))**2)))
SECG = lambda c: lambda a, b: lambda x: (1/cosh((x-a)/(b-a)))
TANGS = lambda c: lambda a, b: lambda x: (1+tanh(-((x-a)/abs(b-a))**2)) ** \
                        log(0.5, (1+tanh(-((c-a)/abs(b-a))**2)))
GAUSS = lambda c: lambda a, b: lambda x: exp(((x-a)**2)*log(0.5) / ((c-a)**2))
CAUCS = lambda c: lambda a, b: lambda x: (1/ (1+ ((x-a) / (c-a))**2))
LOGIS = lambda c: lambda a, b: lambda x: (2/(1+exp(((x-a)/abs(b-a))**2))) ** \
                        log(0.5, ((2/(1+exp(((c-a)/abs(b-a))**2)))))
SECGS = lambda c: lambda a, b: lambda x: (1/cosh((x-a)/abs(b-a))) ** \
                        log(0.5, (1/cosh((c-a)/abs(b-a))))
COSS = lambda c: lambda a, b: lambda x: ((1+cos(pi*(x-a)/abs(b-a)))/2) ** \
                        log(0.5, ((1+cos(pi*(c-a)/abs(b-a)))/2))
POINT = lambda a, b: lambda x: 1

class TrapExt(Subset):
    '''
    Нечеткие числа в обобщенно-трапециевидной форме.
    Синтаксис:
        >>> A = TrapExt(begin = 1.0,
                         end = 4.0,
                         begin_tol = 2.0,
                         end_tol = 3.0,
                         left = LINE(3.0),
                         right = LINE(1.0))

    Первые 4 параметра аналогичны параметрам конструктора трапециевидных
    нечетких чисел (см. Trapezoidal). Два последних определяют форму функций
    скатов - левого и правого. Для этого используется ряд лямбда-выражений,
    включенных в настоящую библиотеку. При их использовании следует
    указать параметр функции - число, характеризующее степень плавности ската.
    Функцию каждого ската выбирает пользователь из следующих альтернатив:
        LINE
            линейный (степенной) скат
        quad
            квадратический (параболический, полукруговой) скат
        lapl
            скат в виде лаплассианы
        tang
            тангенсоида
        gaus
            гауссиана
        cauc
            скат в виде распределения Коши
        logi
            логистическая кривая
        secg
            скат в виде гиперболического секанса
    '''

    def __init__(self, points=(0.0, 1.0, 2.0, 3.0),
                       left=LINE(1),
                       right=LINE(1)):

        (begin, begin_tol, end_tol, end) = points

        super(TrapExt, self).__init__(begin, end)

        self.points["begin"] = begin
        self.points["begin_tol"] = begin_tol
        self.points["end_tol"] = end_tol
        self.points["end"] = end
        self.left = left
        self.right = right
        self.l_skat = left(self.points["begin_tol"], self.points["begin"])
        self.r_skat = right(self.points["end_tol"], self.points["end"])

        if begin == begin_tol:
            self.left = POINT
            self.l_skat = POINT(self.points["begin_tol"], self.points["begin"])
        if end == end_tol:
            self.right = POINT
            self.r_skat = POINT(self.points["end_tol"], self.points["end"])

    def value(self, key):
        if key <= self.points["begin"]:
            return 0.0
        elif self.points["begin"] < key < self.points["begin_tol"]:
            return self.l_skat(key)
        elif self.points["begin_tol"] <= key <= self.points["end_tol"]:
            return 1.0
        elif self.points["end_tol"] < key <= self.points["end"]:
            return self.r_skat(key)
        elif self.points["end"] < key:
            return 0.0
        else:
            return -1

    def fuzziness(self):
        '''
        Возвращает меру нечеткости нечеткого числа
        '''
        tol = self.points["end_tol"] - self.points["begin_tol"]
        supp = self.points["end"] - self.points["begin"]
        lvl = Subset.level(self, 0.5)
        mid = lvl.domain.end - lvl.domain.begin
        return tol*(1-abs(2.0*mid-supp-tol)/(0-tol))

    def _fuzzy_algebra(self, other, operation):
        if isinstance(self, Point) or isinstance(other, Point):
            raise NotImplementedError
        if isinstance(other, float) or isinstance(other, int):
            other = TrapExt((other, other, other, other))

    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            other = TrapExt((other, other, other, other))

        res_begin = min(self.points["begin"] + other.points["begin"],
                        self.points["begin"] + other.points["end"],
                        self.points["end"] + other.points["begin"],
                        self.points["end"] + other.points["end"])
        res_begin_tol = min(self.points["begin_tol"]+other.points["begin_tol"],
                            self.points["begin_tol"] + other.points["end_tol"],
                            self.points["end_tol"] + other.points["begin_tol"],
                            self.points["end_tol"] + other.points["end_tol"])
        res_end_tol = max(self.points["begin_tol"]+other.points["begin_tol"],
                            self.points["begin_tol"]+other.points["end_tol"],
                            self.points["end_tol"]+other.points["begin_tol"],
                            self.points["end_tol"]+other.points["end_tol"])
        res_end = max(self.points["begin"]+other.points["begin"],
                        self.points["begin"]+other.points["end"],
                        self.points["end"]+other.points["begin"],
                        self.points["end"]+other.points["end"])

        res_left = lambda x, y: lambda z: \
                self.left(self.points["begin_tol"], self.points["begin"])\
                (self.points["begin"]+(self.points["begin_tol"]-\
                self.points["begin"])*\
                (z-res_begin)/(res_begin_tol-res_begin)) * \
                other.left(other.points["begin_tol"], other.points["begin"])\
                (other.points["begin"]+(other.points["begin_tol"]-\
                other.points["begin"])*\
                (z-res_begin)/(res_begin_tol-res_begin))
        res_right = lambda x, y: lambda z: \
                self.right(self.points["end_tol"], self.points["end"])\
                (self.points["end_tol"]+(self.points["end"]-\
                self.points["end_tol"])*\
                (z-res.end_tol)/(res_end-res_end_tol)) * \
                other.right(other.points["end_tol"], other.points["end"])\
                (other.points["end_tol"]+(other.points["end"]-\
                other.points["end_tol"])*\
                (z-res_end_tol)/(res_end-res_end_tol))

        res = TrapExt(begin=res_begin, begin_tol=res_begin_tol,
                        end_tol=res_end_tol, end=res_end,
                        left=res_left, right=res_right)
        return res

    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            other = TrapExt(begin=other, end=other,
                                begin_tol=other, end_tol=other)

        res_begin = min(self.points["begin"]-other.points["begin"],
                        self.points["begin"]-other.points["end"],
                        self.points["end"]-other.points["begin"],
                        self.points["end"]-other.points["end"])
        res_begin_tol = min(self.points["begin_tol"]-other.points["begin_tol"],
                            self.points["begin_tol"]-other.points["end_tol"],
                            self.points["end_tol"]-other.points["begin_tol"],
                            self.points["end_tol"]-other.points["end_tol"])
        res_end_tol = max(self.points["begin_tol"]-other.points["begin_tol"],
                            self.points["begin_tol"]-other.points["end_tol"],
                            self.points["end_tol"]-other.points["begin_tol"],
                            self.points["end_tol"]-other.points["end_tol"])
        res_end = max(self.points["begin"]-other.points["begin"],
                        self.points["begin"]-other.points["end"],
                        self.points["end"]-other.points["begin"],
                        self.points["end"]-other.points["end"])

        res_left = lambda x, y: lambda z: self.left(self.points["begin_tol"],
                self.points["begin"])\
                (self.points["begin"]+(self.points["begin_tol"]-\
                self.points["begin"])*\
                (z-res_begin)/(res_begin_tol-res_begin)) * \
                other.left(other.points["begin_tol"], other.points["begin"])\
                (other.points["begin"]+(other.points["begin_tol"]-\
                other.points["begin"])*\
                (z-res_begin)/(res_begin_tol-res_begin))
        res_right = lambda x, y: lambda z: self.right(self.points["end_tol"],
                self.points["end"])\
                (self.points["end_tol"]+(self.points["end"]-\
                self.points["end_tol"])*\
                (z-res_end_tol)/(res_end-res_end_tol)) * \
                other.right(other.points["end_tol"], other.points["end"])\
                (other.points["end_tol"]+(other.points["end"]-\
                other.points["end_tol"])*\
                (z-res_end_tol)/(res_end-res_end_tol))

        res = TrapExt(begin=res_begin, begin_tol=res_begin_tol,
                        end_tol=res_end_tol, end=res_end,
                        left=res_left, right=res_right)
        return res

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            other = TrapExt(begin=other, end=other,
                                begin_tol=other, end_tol=other)

        res_begin = min(self.points["begin"]*other.points["begin"],
                        self.points["begin"]*other.points["end"],
                        self.points["end"]*other.points["begin"],
                        self.points["end"]*other.points["end"])
        res_begin_tol = min(self.points["begin_tol"]*other.points["begin_tol"],
                            self.points["begin_tol"]*other.points["end_tol"],
                            self.points["end_tol"]*other.points["begin_tol"],
                            self.points["end_tol"]*other.points["end_tol"])
        res_end_tol = max(self.points["begin_tol"]*other.points["begin_tol"],
                            self.points["begin_tol"]*other.points["end_tol"],
                            self.points["end_tol"]*other.points["begin_tol"],
                            self.points["end_tol"]*other.points["end_tol"])
        res_end = max(self.points["begin"]*other.points["begin"],
                        self.points["begin"]*other.points["end"],
                        self.points["end"]*other.points["begin"],
                        self.points["end"]*other.points["end"])

        res_left = lambda x, y: lambda z: \
                        self.left(x, y)(z)*other.left(x, y)(z)
        res_right = lambda x, y: lambda z: \
                        self.right(x, y)(z)*other.right(x, y)(z)

        res = TrapExt(begin=res_begin, begin_tol=res_begin_tol,
                        end_tol=res_end_tol, end=res_end,
                        left=res_left, right=res_right)
        return res

    def __div__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            other = TrapExt(begin=other, end=other,
                                begin_tol=other, end_tol=other)

        res_begin = min(self.points["begin"]/other.points["begin"],
                        self.points["begin"]/other.points["end"],
                        self.points["end"]/other.points["begin"],
                        self.points["end"]/other.points["end"])
        res_begin_tol = min(self.points["begin_tol"]/other.points["begin_tol"],
                            self.points["begin_tol"]/other.points["end_tol"],
                            self.points["end_tol"]/other.points["begin_tol"],
                            self.points["end_tol"]/other.points["end_tol"])
        res_end_tol = max(self.points["begin_tol"]/other.points["begin_tol"],
                            self.points["begin_tol"]/other.points["end_tol"],
                            self.points["end_tol"]/other.points["begin_tol"],
                            self.points["end_tol"]/other.points["end_tol"])
        res_end = max(self.points["begin"]/other.points["begin"],
                        self.points["begin"]/other.points["end"],
                        self.points["end"]/other.points["begin"],
                        self.points["end"]/other.points["end"])

        res_left = lambda x, y: lambda z: \
                        self.left(x, y)(z)*other.left(x, y)(z)
        res_right = lambda x, y: lambda z: \
                        self.right(x, y)(z)*other.right(x, y)(z)

        res = TrapExt((res_begin, res_begin_tol, res_end_tol, res_end),
                        left=res_left, right=res_right)
        return res

    # TODO radd, rsub, rmul, rdiv

class TrFN(TrapExt):
    '''
    Трапециевидное нечеткое число
    '''

    def __init__(self, a, b, c, d):
        TrapExt.__init__(self, (a, b, c, d), LINE(1), LINE(1))
##        super(TrFN, self).__init__(a, b, c, d, LINE(1), LINE(1))

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    #~ doctest.testmod(verbose=True)
