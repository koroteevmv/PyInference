# coding=utf-8

from pyinference.fuzzy.tnorm import MinMax
from pyinference.fuzzy.subset import Interval
from pyinference.fuzzy.set import Partition
from pyinference.fuzzy.domain import RationalRange
##from fuzzy.aggregation import Simple, Tree, Rules


class AggregationMetod(object):
    def calculate(self, host):
        pass


class Simple(AggregationMetod):
    def calculate(self, host):
        est = 0.0
        weight = 0.0
        for child in host.childs.values():
            try:
                est += float(child.get_estim())
            except TypeError:
                return None
            weight += 1.0
        if weight == 0.0:
            return None
        else:
            host.estimation = est/weight
            return host.estimation


class Rules(AggregationMetod):
    def __init__(self):
        self.rules = []

    def add_rule(self, ant=None, concl='', name=''):
        if not ant:
            ant = {}
        self.rules.append(Rule(ant=ant, concl=concl, name=name))


class Mamdani(Rules):
    def __init__(self, cond=None, cons=None, tnorm=None):
        super(Mamdani, self).__init__()
        self.cond = cond or []
        self.cons = cons or []
        self.tnorm = tnorm or MinMax()

    def calculate(self, values):
        res = Interval(self.cons.domain.begin,
                       self.cons.domain.end) * 0.0
        for rule in self.rules:
            alpha = 1.0
            for i in xrange(len(values)):
                fact_class_name = rule.ant[i]
                mem = self.cond[i].sets[fact_class_name].value(values[i])
                alpha = self.tnorm.norm(alpha, mem)
            rule.alpha = alpha
            res = res | (Interval(self.cons.domain.begin,
                                self.cons.domain.end) * rule.alpha & self.cons[rule.concl])
        return res

class RulesAccurate(Rules):
    def calculate(self, host):
        sum_a = 0.0
        summ = 0.0
        # для каждого правила вычисляем его альфу
        for rule in self.rules:
            alpha = 1.0 # Для t-нормы начальным значением будет 1
            # для каждого фактора в правиле
            for param, value in rule.ant.iteritems():
                # значение фактора
                fact = host[param].get_estim()
                # его принадлежность в классификаторе
                mem = host[param].classifier[value].value(fact)
                alpha = host.tnorm.t_norm(alpha, mem)
            rule.alpha = alpha
            sum_a += alpha
            summ += host.classifier[rule.concl].centr()*alpha
        if sum_a == 0:
            return 0.0
        return summ/sum_a

class Rule(object):
    def __init__(self, ant=None, concl='', name=''):
        if not ant:
            ant = {}
        self.concl = concl
        self.name = name
        self.ant = ant

    def __str__(self):
        res = str(self.name)+': '
        for (name, value) in self.ant.iteritems():
            res += str(name)+'='+value+' '
        res += ' -> '+str(self.concl)
        return res


def main():
    import pylab as py
    import numpy as np
    from matplotlib.widgets import MultiCursor

    domain01 = RationalRange(acc=10)
    IS = Partition(domain=domain01, peaks=[0.0, 0.5, 1.0])
    ST = Partition(domain=domain01, peaks=[0.0, 0.5, 1.0])
    LQ = Partition(domain=domain01, peaks=[0.0, 0.25, 0.5, 0.75, 1.0])

##    IS.plot()
##    py.show()

    infer = Mamdani(cond=[IS, ST], cons=LQ)
    infer.add_rule(ant=['0', '0'], concl='0', name='first')
    infer.add_rule(ant=['2', '2'], concl='4', name='first')
##    print infer.cond

##    res = infer.calculate1([0.0, 0.5])
##    print res.centr()
##    res.plot()
##
##    py.show()

    xs = np.arange(0.0, 1.0, 0.2)
    z = []
    for x in xs:
        line=[]
        for y in xs:
            line.append(infer.calculate([x, y]).centr())
        z.append(line)
    print z

    fig = py.figure()

    inf = fig.add_subplot(122)
    c1 = fig.add_subplot(221)
    c2 = fig.add_subplot(223)
    inf.contourf(xs, xs, z)
    IS.plot(subplot=c1)
    ST.plot(subplot=c2)

    cursor = MultiCursor(fig.canvas, (inf, c1), color='white')

    py.show()


if __name__ == '__main__':
    main()
