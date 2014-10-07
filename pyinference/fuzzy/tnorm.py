# -*- coding: UTF-8 -*-

'''
Модуль реализует набор простых и параметрических треугольный норм и конорм.
'''

import math


class Tnorm(object):
    def norm(self, i, j):
        pass

    def conorm(self, i, j):
        pass


class MinMax(Tnorm):
    def norm(self, i, j):
        return min(i, j)

    def conorm(self, i, j):
        return max(i, j)


class SumProd(Tnorm):
    def norm(self, i, j):
        return i*j

    def conorm(self, i, j):
        return i+j-i*j



class Margin(Tnorm):
    def norm(self, i, j):
        return max(i+j-1, 0)

    def conorm(self, i, j):
        return min(i+j, 1)


class Drastic(Tnorm):
    def norm(self, i, j):
        if i == 1:
            return j
        elif j == 1:
            return i
        else:
            return 0

    def conorm(self, i, j):
        if i == 0:
            return j
        elif j == 0:
            return i
        else:
            return 1


class ParametricNorm(Tnorm):
    def __init__(self, param):
        self.param = param


class Tnorm1(ParametricNorm):
    def __init__(self, param):
        super(Tnorm1, self).__init__(param)
    def norm(self, i, j):
        return i*j/(self.param+(1-self.param)*(i+j-i*j))
    def conorm(self, i, j):
        return (i+j-(2-self.param)*i*j)/(1-(1-self.param)*i*j)


class Tnorm2(ParametricNorm):
    def __init__(self, param):
        super(Tnorm2, self).__init__(param)
    def norm(self, i, j):
        return i*j / max(i, j, self.param)
    def conorm(self, i, j):
        return (i+j-i*j-min(i, j, 1-self.param)) / max(1-i, 1-j, self.param)


class Tnorm3(ParametricNorm):
    def __init__(self, param):
        super(Tnorm3, self).__init__(param)
    def norm(self, i, j):
        try:
            return 1/(1+ ((1/i - 1)**self.param +
                    (1/j - 1)**self.param)**(1/self.param))
        except ZeroDivisionError:
            return 0.0
    def conorm(self, i, j):
        return 1/(1+ ((1/i - 1)**-self.param + \
                    (1/i - 1)**-self.param)**(1/self.param))


class Tnorm4(ParametricNorm):
    def __init__(self, param):
        super(Tnorm4, self).__init__(param)
    def norm(self, i, j):
        return 1 - ((1-i)**self.param+(1-j)**self.param-(1-i) ** \
                        self.param*(1-j)**self.param)**(1/self.param)
    def conorm(self, i, j):
        return (i**self.param + j**self.param - i**self.param*j ** \
                        self.param)**(1/self.param)


class Tnorm5(ParametricNorm):
    def __init__(self, param):
        super(Tnorm5, self).__init__(param)
    def norm(self, i, j):
        return max((1 - ((1-i)**self.param + (1-j)**self.param) ** \
                        (1/self.param)), 0)
    def conorm(self, i, j):
        return min((i**self.param + j**self.param), 1)


class Tnorm6(ParametricNorm):
    def __init__(self, param):
        super(Tnorm6, self).__init__(param)
    def norm(self, i, j):
        try:
            return math.log((1+ (self.param**i -1)*(self.param**j -1) / \
                                (self.param-1)), self.param)
        except ZeroDivisionError:
            return 0.0
    def conorm(self, i, j):
        return 1 - math.log((1+ (self.param**(1-i) + self.param**(1-j)) / \
                   (self.param-1)), self.param)


class Tnorm7(ParametricNorm):
    def __init__(self, param):
        super(Tnorm7, self).__init__(param)
    def norm(self, i, j):
        return max((i+j-1+self.param*i*j)/(1+self.param), 0)
    def conorm(self, i, j):
        return min((i+j-1+self.param*i*j), 1)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    #~ doctest.testmod(verbose=True)
