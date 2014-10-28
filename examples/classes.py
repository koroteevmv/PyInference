from pyinference.fuzzy.domain import RationalRange, Domain


class Variable(object):
    def __init__(self, domain=None, name=''):
        self.domain = domain or RationalRange()
        self.name = name
        self.value = None

    def __eq__(self, other):
        raise NotImplementedError

    def get_value(self):
        return self.value

    def set_value(self, value):
        assert value in self.domain
        self.value = value

    def card(self):
        return self.domain.card()


class Node(object):
    def __init__(self):
        self.variable = None
        self.parents = []
        self.name = ''


class RandomVariable(Variable):
    pass


class LinguisticVariable(Variable):
    pass


class InferenceRule(object):
    def __init__(self, cond, cons, name=''):
        self.cond = cond
        self.cons = cons
        self.name = name

    pass


class MixedNode(Node):
    def __init__(self, var, parents=None, name=''):
        super(MixedNode, self).__init__()
        self.variable = var
        self.parents = parents or []
        self._parents_table = {}
        for var in parents:
            self._parents_table[var.name] = var
        self.name = name
        self.rules = []
        self.card = 1
        self.pcard = []
        self.kcard = []
        for var in self.parents:
            self.pcard.append(var.card())
            self.card *= var.card()
        x = 1
        for i in self.pcard[::-1]:
            x *= i
            self.kcard.append(x)
        self.kcard.insert(0, 1)
        self.kcard = self.kcard[1::-1]

    def add_rule(self, cond, cons, name=''):
        # assertions
        self.rules.append(InferenceRule(cond=cond, cons=cons, name=name))

    def calculate(self):
        if self.variable.value:
            return self.variable.value
        else:
            for rule in self.rules:
                applies = True
                for name, value in rule.cond.iteritems():
                    applies = applies and (self._parents_table[name].value == value)
                if applies:
                    return rule.cons

    def cond_table(self):
        res = []
        for i in xrange(self.card):
            assingment = self._ind2ass(i)
            res.append(assingment)
        return res

    def _ind2ass(self, i):
        assingment = []
        for (k, j) in zip(self.pcard, self.kcard):
            assingment.append((i / j) % k)
        assingment_ = []
        for (ind, var) in zip(assingment, self.parents):
            assingment_.append(var.domain[ind])
        return assingment_

    def cons_table(self):
        for rule in self.rules:
            print rule.name, ":", rule.cond, "->", rule.cons
        pass

    def apply_rule_table(self, table):
        if len(table) != self.card:
            raise AttributeError
        for i in xrange(len(table)):
            ass = self._ind2ass(i)
            cond = {}
            for (var, val) in zip(self.parents, ass):
                cond[var.name] = val
            self.add_rule(cond=cond, cons=table[i], name=str(i))


class LogicNode(MixedNode):
    def __init__(self, var, parents=None, name=''):
        super(LogicNode, self).__init__(var)
        self.variable = var
        self.parents = parents or []
        self._parents_table = {}
        for var in parents:
            self._parents_table[var.name] = var
        self.name = name
        self.rules = []
        self.card = 1
        self.pcard = []
        self.kcard = []
        for var in self.parents:
            self.pcard.append(var.card())
            self.card *= var.card()
        x = 1
        for i in self.pcard[::-1]:
            x *= i
            self.kcard.append(x)
        self.kcard.insert(0, 1)
        self.kcard = self.kcard[1::-1]

    def calculate(self):
        if self.variable.value:
            return self.variable.value
        else:
            for rule in self.rules:
                applies = True
                for name, value in rule.cond.iteritems():
                    applies = applies and (self._parents_table[name].value == value)
                if applies:
                    return rule.cons

    pass


class DeterministicNode(Node):
    pass


class TermSet(Domain):
    def __init__(self, terms=None):
        super(TermSet, self).__init__()
        self.terms = terms or []

    def __iter__(self):
        for i in self.terms:
            yield i

    def __contains__(self, x):
        return x in self.terms

    def card(self):
        return len(self.terms)

    def __getitem__(self, key):
        return self.terms[key]


# ###############################################################################


##from fuzzy.set import Partition, FuzzySet

binary = TermSet(['low', 'high'])
ternary = TermSet(['low', 'middlle', 'high'])

a = Variable(domain=binary, name='A')
b = Variable(domain=binary, name='B')
c = Variable(domain=binary, name='C')

B = LogicNode(b, parents=[a, c], name='B|A,C')

#######################################

##B.add_rule(cond={'A':'low'}, cons='low', name='first')
##B.add_rule(cond={'A':'high', 'C':'low'}, cons='low', name='second')
##B.add_rule(cond={'A':'high', 'C':'high'}, cons='high', name='second')
##
##a.set_value('high')
##c.set_value('high')
##
##print a.get_value()
##print c.get_value()
##print B.calculate()

#######################################

print B.cond_table()
B.apply_rule_table(['low', 'low', 'low', 'high'])
B.cons_table()
print

a.set_value('high')
c.set_value('high')

print a.get_value()
print c.get_value()
print B.calculate()