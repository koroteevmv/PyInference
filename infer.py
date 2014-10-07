# -*- coding: UTF-8 -*-

'''Модуль для работы с механизмом нечеткого вывода.

Позволяет строить простейшие нечеткие экспертные системы (FES) путем построения
нечетких подмножеств на деревьях особого типа.
Синтаксис:
    >>> from FuzzySet import *
    >>> # создаем классификаторы:
    >>> C1=std_5_Classificator(name='Classifier 1')
    >>> C2=std_3_Classificator(name='Classifier 2', begin=0.0, end=100.0)
    >>> C3=std_2_Classificator(name='Classifier 3', begin=10.0, end=30.0)
    >>> # создаем дерево
    >>> T=Ruled(name='Rule system', clas=C1)
    >>> T.add(Ruled(name='Factor 1', clas=C2))
    >>> T.add(Ruled(name='Factor 2', clas=C3))
    >>> # добавляем правила
    >>> T.add_rule({'Factor 1':'I',   'Factor 2':'I', }, concl='I'  )
    >>> T.add_rule({'Factor 1':'I',   'Factor 2':'II',}, concl='II' )
    >>> T.add_rule({'Factor 1':'II',  'Factor 2':'I', }, concl='II' )
    >>> T.add_rule({'Factor 1':'II',  'Factor 2':'II',}, concl='III')
    >>> T.add_rule({'Factor 1':'III', 'Factor 2':'I', }, concl='III')
    >>> T.add_rule({'Factor 1':'III', 'Factor 2':'II',}, concl='III')
    >>> # добавляем оценки
    >>> T['Factor 1'].set_estim(6.5)
    >>> T['Factor 2'].set_estim(12.6)
    >>> print T.get_estim()
    9.55
    >>> # получаем результат с использованием разных t-норм
    >>> T.tnorm=min_max()
    >>> T.calculate()  #doctest: +SKIP
    <FuzzySubset.Subset instance at 0x0292EF80>
    >>> print T.get_estim()
    9.55
    >>> T.tnorm=sum_prod()
    >>> T.calculate()  #doctest: +SKIP
    <FuzzySubset.Subset instance at 0x02B3A3F0>
    >>> print T.get_estim()
    9.55
    >>> T.tnorm=margin()
    >>> T.calculate()  #doctest: +SKIP
    <FuzzySubset.Subset instance at 0x02B3A788>
    >>> print T.get_estim()
    9.55
'''

from .subset import Interval
from .domain import Domain
from .tnorm import MinMax
from .aggregation import Simple, Tree, Rules

class AggregationMetod(object):
    '''
    Класс определяет интерфейс к различным методам агрегации частных показателей
    в интегральный. Подклассы данного класса реализуют алгоритмы интеграции
    различных типов нечетких контроллеров.
    '''
    def calculate(self, host):
        '''
        Метод возвращает агрегированное значение
        '''
        pass

class Simple(AggregationMetod):
    '''
    Метод агрегации показателей, в котором интегральный показатель расчитывается
    как среднее арифметическое частных.
    '''
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
    '''
    Данный класс объединяет группу методов расчета интегральных показетелей,
    основанных на использовании системы нечетких правил. Все типы нечеткого
    вывода, использующие правила вывода реализуются классами, дочерними от
    данного.
    '''
    def __init__(self):
        self.rules = []

    def add_rule(self, ant=None, concl='', name=''):
        '''
        Данный метод позволяет добавить систему правил, согласно которой будет
        вычисляться оценка текущего узла дерева (к которому применен метод) в
        зависимости от оценок его потомков.
        Нечеткое правило состоит из посылки и заключения. Посылка описывает
        при каких значениях факторов результирующая оценка принимает значение,
        описанное в заключении. В посылке перечислены имена факторов и имена
        термов соответствующих им классификаторов; а в заключении - имя терма
        результирующего параметра.
        Синтаксис:
            >>> T=Ruled()
            >>> T.add_rule(name='rule 1',
                            ant={'factor': 'value'},
                            concl='value')  #doctest: +SKIP

        Параметры:
            ant
                Посылка нечеткого правила.
                Ассоциативный массив, в ключах которого задаются имена факторов,
                а в значениях - соответствующие имена значений лингвистической
                переменной. Предполагается, что в данном массиве перечислены
                посылки нечеткого правила, связанные логическим И. Для ввода
                правил, в посылке которых встречается союз ИЛИ используйте
                разбиение на несколько правил.
            concl
                Заключение нечеткого правила
                Имя терма классификатора, соответствующее данной посылке.
            name
                имя правила, используемое опционально для удобства.
        '''
##        # проверяем првильность параметров:
##        for param, value in ant.iteritems():
##            # param должен быть среди потомков
##            host[param]
##            # value должен быть среди термов его классификатора
##            host[param].classifier[value]
##        # concl должен быть среди термов собственного классификатора
##        host.classifier[concl]
        if not ant:
            ant = {}
        self.rules.append(Rule(ant=ant, concl=concl, name=name))

class Mamdani(Rules):
    '''
    Данный класс реализует функциональность контроллера Мамдани, то есть
    классический алгоритм нечеткого композитного вывода.

    Для каждого правила из системы правил считается его вес: как t-норма
    принадлежности фактических значений показателей термам, упомянутым в
    посылке правила. Затем терм, соответствующий заключению правила обрезается
    на уровне, равном весу показателей.
    Таким образом, для каждого правила выводится НПМ на области определения
    результирующего показателя. Они объединяются путем применения к ним
    t-конормы и получившееся НПМ и бдет являться конечным результатом процесса
    нечеткого вывода.
    '''
    def __init__(self):
        super(Mamdani, self).__init__(self)

    def calculate(self, host):
        # Начальное значение итогового НПМ. Для конормы это 0 уровень
        res = Interval(host.classifier.begin,
                       host.classifier.end,
                       tnorm=host.tnorm) * 0.0
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
            # обрезаем терм собственного классификатора уровнем альфа
            # и прибавляем его к существующим, используя конорму
            res = res.t_conorm(Interval(host.classifier.begin,
                                         host.classifier.end) * rule.alpha & \
                                         host.classifier[rule.concl])
        return res

class RulesAccurate(Rules):
    '''
    Данный алгоритм нечеткого вывода в общем аналогичен контроллеру Мамдани,
    однако, результат определяется как среднне арифметическое взвешенное по
    дефаззифицированным термам заключения каждого правила, причем весами
    являются веса соответствующего правила.
    '''
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
    '''
    Описание
    Синтаксис:
        >>>
    Attributes:
        ant
        concl
        name
    '''
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

class Tree(Domain):
    '''
    Представляет собой носитель нечеткого множества в виде иерархической
    структуры (дерева), в которой оценка данного узла зависит определенным
    образом от оценок его потомков.
    Конструктор данного класса создает как само дерево, так и его потомков.
    Листовой элемент дерева - это тот, для которого не создано ни одного
    потомка.
    Синтаксис:
        >>> A=Tree('tree')
        >>> A.add(Tree('branch 1'))
        >>> A.add(Tree('branch 2'))
        >>> A.add(Tree('branch 3'))
        >>> A['branch 1'].add(Tree('branch 1 1'))
        >>> A['branch 1'].add(Tree('branch 1 2'))
        >>> A['branch 1'].add(Tree('branch 1 3'))
        >>> A['branch 1']['branch 1 2'].add(Tree('leaf 1 2 1'))
        >>> A['branch 1']['branch 1 2'].add(Tree('leaf 1 2 2'))
        >>> A['branch 1']['branch 1 2'].add(Tree('leaf 1 2 3'))
        >>> A.char()
        branch 1 3 - None (1.0)
        leaf 1 2 1 - None (1.0)
        leaf 1 2 3 - None (1.0)
        leaf 1 2 2 - None (1.0)
        branch 1 2 - None (1.0)
        branch 1 1 - None (1.0)
        branch 1 - None (1.0)
        branch 2 - None (1.0)
        branch 3 - None (1.0)
        tree - None (1.0)
        >>> A=Tree('name', estim=2.5, weight=0.23, clas=std_3_Classifier(),
                    tnorm=sum_prod)

    Параметры:
        name
            задает имя узла, по которому к нему можно будет обращаться
        estim
            степень принадлежности данного узла
        clas
            классификатор, используемый для оценки уровня данного параметра
        tnorm
            задает используемые при интеграции t-нормы и кономры. Подробнее см.
            FuzzyDomain.t_norm
        agg
            метод аггрегации частных показателей в интегральный.
            См. FuzzyDomain.AggregationMethod
    Переменные класса:
        childs
        name
        estimation
        weight
        classifier
        tnorm
    '''
    # TODO отделить МАИ от иерархического носителя
    # TODO реализовать в интерфейсе Subset иерархический носитель.
    # Без изъебов типа весов и классификаторов. Но с A.value()

    def __init__(self, name='', estim=None, agg=Simple(),
                        clas=None, tnorm=MinMax()):
        self.name = name
        self.estimation = estim
        self.childs = {}
        self.agg = agg
        self.classifier = clas
        self.tnorm = tnorm

    def __str__(self):
        '''
        Для быстрого вывода основной информации о дереве, поддереве или листе,
        можно использовать процедуру преобразования к строковому типу.
        Синтаксис:
            >>> T=Tree('tree')
            >>> T.add(Tree('branch 1', 2))
            >>> T.add(Tree('branch 2', 3))
            >>> print T
            tree - 2.5 (1.0)
            >>> print T['branch 1']
            branch 1 - 2 (1.0)

        Данный синтаксис можно комбинировать с синтаксисом __iter__ для вывода
        более полной информации о всех узлах дерева:
            >>> for i in T:
            ...     print i
            ...

        '''
        return self.name+' - '+ str(self.get_estim())

    def __iter__(self):
        '''
        Для быстрого перебора всех дочерних элементов дерева можно использовать
        объект данного класса как итератор. Порядок, в котором возвращаются
        узлы дерева соответствует алгоритму postorder traversal, то есть
        сначала перечисляются все дочерние узлы, затем родительский узел. и так
        для каждого узла, начиная с вершины.
        Синтаксис:
            >>> T=Tree('tree')
            >>> T.add(Tree('branch 1', 2))
            >>> T.add(Tree('branch 2', 3))
            >>> for i in T:
            ...     print i.name
            ...
            branch 1
            branch 2
            tree
        '''
        for leaf in self.childs.values():
            for i in leaf:
                yield i
        yield self

    def add(self, addition):
        '''
        Описание
        Синтаксис:
        Параметры:
            Параметр
                описание
        '''
        self.childs[addition.name] = addition

    def get_estim(self):
        '''
        Описание
        Синтаксис:
        Параметры:
            Параметр
                описание
        '''
        if self.estimation or self.estimation == 0.0:
            return self.estimation
        else:
            if self.childs == []:
                return None
            return self.agg.calculate(self)

    def set_estim(self, val):
        '''
        Описание
        Синтаксис:
        Параметры:
            Параметр
                описание
        '''
        self.estimation = val

    def __getitem__(self, param):
        '''
        Для быстрого доступа к любому из дочерних узлов дерева (не обязательно
        прямых потомков) по названию можно использовать следующий синтаксис:
            >>> T=Tree('tree')
            >>> T.add(Tree('branch 1', 2))
            >>> T.add(Tree('branch 2', 3))
            >>> T.char()
            branch 1 - 2 (1.0)
            branch 2 - 3 (1.0)
            tree - 2.5 (1.0)
            >>> print T['branch 1']
            branch 1 - 2 (1.0)
        '''
##        print 'im here!'
        return self.childs[param]


# XXX интерфейс FES с модельными параметрами и возможностью задания
# пользовательских и изменения на лету.

from .aggregation import Simple, Tree, Rules
from .tnorm import MinMax


class Controller(object):
    '''
    Данный класс представляет интерфейс для создания нечеткого контроллера со
    множественными входами и выходами. Входом нечеткого контроллера называется
    лингвистическая переменная, имеющая имя и множество терм-значений
    (классификатор), которой присваивается четкое, нечеткое или лингвистическое
    значение. Выходом контроллера называется лингвистическая переменная,
    имеющая имя и множество терм-значений, значение которой рассчитывается,
    исходя из значений входных переменных по определенному алгоритму, который
    называется тип контроллера (см. FuzzyDomain.AggregationMethod).

    Синтаксис:
        >>>
    Параметры конструктора:
        input_
            C помощью этого параметра задаются входные переменные контроллера.
            В этот параметр следует передать ассоциативный массив, ключами
            которого являются строки-имена входных переменных, а значениями -
            соответствующие классификаторы, задающие терм-множество каждой
            переменной.
        out
            Подобным же образом задаются и выходные переменные классификатора.
        rules
            В данный параметр передаются нечеткие правила вывода (если они
            требуются). Следует передать массив, в котором каждый элемент это
            правило, представленное в виде пары (tuple) посылки и заключения,
            каждая из которых представлена в виде ассоциативного массива с
            ключами - именами переменны и значениями - именами термов.
        method
            Данный параметр определяет тип контроллера. Фактически он задает
            метод (алгоритм) сводки частных показателей в интегральный. Может
            принимать в качестве значения имя любого подкласса
            FuzzyDomain.AggregationMethod.
        tnorm
            Определяет пару треугольных норм и конорм
    Поля класса:
        trees
            Ассоциативный массив, ключами которого являются имена выходных
            переменных контроллера, а значениями - соответствующие им деревья
            (см. FuzzyDomain.Tree)
        inputs
            Ассоциативный массив, ключами которого являются имена входных
            переменных контроллера, а значениями - соответствующие им множества
            терм-значений (классификаторы). См. FuzzySet.FuzzySet
        method
        tnorm
    '''

    def __init__(self, input_=None,
                        out=None,
                        method=Simple(),
                        tnorm=MinMax()):
        '''
        Описание
        Синтаксис:
            >>>
        '''

        if not input_:
            input_ = {}
        if not out:
            out = {}

        self.method = method
        self.tnorm = tnorm
        self.trees = {}
        self.inputs = {}

        self.define_input(input_)
        self.define_output(out)

    def _char(self):
        '''
        Функция выводит данные о нечетком контроллере в человеко-читаемом виде.
        '''
        print "<Fuzzy controller>"
        for name in self.trees.itervalues():
            for i in name:
                print i
            print
        print 'Rules:'
        for tree in self.trees.itervalues():
            if isinstance(tree.agg, Rules):
                for rule in tree.agg.rules:
                    print rule
        print
        print '</Fuzzy controller>'

    def define_input(self, input_):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        self.inputs = {}
        for name in input_.iterkeys():
            self.inputs[name] = Tree(name=name, clas=input_[name])
        return self

    def define_output(self, out):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        self.trees = {}
        for name in out.iterkeys():
            tree = Tree(name=name,
                     clas=out[name],
                     agg=self.method(),
                     tnorm=self.tnorm)
            for branch in self.inputs.itervalues():
                tree.add(branch)
            self.trees[name] = tree
        return self

    def add_input(self, name, clas):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        self.inputs[name] = Tree(name=name, clas=clas)

    def add_output(self, name, clas):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        tree = Tree(name=name, clas=clas, agg=self.method(), tnorm=self.tnorm)
        for branch in self.inputs.itervalues():
            tree.add(branch)
        self.trees[name] = tree

    def define_rules(self, rules):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        if isinstance(self.method(), rules):
            i = 0
            for rule in rules:
                ant, conc = rule
                for name in conc.iterkeys():
                    self.trees[name].agg.add_rule(name='rule '+str(i),
                                                    ant=ant,
                                                    concl=conc[name])
                    i += 1
            return self

    def add_rule(self, rule, name=''):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        # TODO
        pass

    def set(self, input_values):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        for name in input_values.iterkeys():
            self.inputs[name].set_estim(input_values[name])

    def get(self):
        '''
        Описание
        Синтаксис:
            >>>
        '''
        res = {}
        for tree in self.trees.itervalues():
            res[tree.name] = tree.get_estim()
        return res

        #TODO вывод классификаторов входов
        #TODO вывод классификаторов выходов
        #TODO вывод двумерных графиков

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    #~ doctest.testmod(verbose=True)
