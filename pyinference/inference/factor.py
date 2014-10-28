# coding=utf-8

from numpy import repeat, ones, array
from pyinference.inference.variable import Variable

__author__ = 'sejros'


def _itershape(tup):
    """ Итерирование по кортежу.
    """
    r = array(tup).prod()
    res = []
    for i in range(r):
        row = []
        index = i + 0
        for j in range(len(tup)):
            row.insert(0, index % tup[-j - 1])
            index = (index - (index % tup[-j - 1])) / tup[-j - 1]
        res.append(row)
    return res


class Factor(object):
    """ Фактор логического вывода.

    Фактор связывает несколько переменных в совместное распределение вероятности.

    Фактор это некая функция, которая ставит в соответствие каждому возможному набору значений некого множества
    переменных действительное число (значение фактора). Набор соответствующих значений A некоторого множества
    переменных V называется назначением (assignment), если:
    - мощность множества A равна мощности множества V;
    - каждый элемент множества A является элементом множества значений соответствующего элемента множества V

    Пример: допустим, M – бинарная переменная M=(“монета”, {“орел”, “решка”}), а V – набор переменных V={M}3={M, M, M},
    тогда А={“орел”, “орел”, “решка”} – корректное назначение, а А1={“решка”, “решка”} – некорректное назначение,
    так как не указано значение третьей переменной. А2={“решка”, “решка”, “ребро”} – также не является корректным
    назначением, так как третий элемент («ребро») отсутствует в области определения третьей переменной M. Однако,
    назначение А2 является корректным для множества V’={M, M, M’}, где M’=(“другая монета”, {“орел”, “решка”, “ребро”}),
    так как теперь третья переменная во множестве V’ уже другая и она содержит значение «ребро» в своей области
    определения. По тем же причинам, назначение А2 является корректным для набора V’’={M’, M’, M’}.

    Таким образом, фактор определяется упорядоченной тройкой F=(N, V, X), где N – имя фактора, V – область определения
    фактора (набор переменных, на которых определен фактор), X – множество значений фактора соответствующей мощности.

    Фактор определен на некотором упорядоченном множестве переменных, называемом областью определения (scope) фактора.
    Область определения обозначается так: F(A,B,C) или так: scope(F)={A,B,C}, где F – фактор, A, B и C – переменные,
    входящие в фактор. Заметим, что определение фактора не накладывает никаких ограничений ни на отдельные значения,
    ни на сумму этих значений, несмотря на то, что теория вероятности предусматривает такие ограничения. Фактор – это
    базовый строительный блок в вероятностных графических моделях. и представление вероятностей – всего лишь одно из
    возможных применений факторов. Например, определим фактор на одной переменной M, следующим образом::

        coin_var = Variable(name="coin", terms=["орел", "решка"])
        coin_factor = Factor(name="coin", cons=[coin_var])
        import numpy as np
        coin_factor.cpd = np.array([0.5, 0.5])

    .. note::
        Заметим, что по умолчанию, распределение фактора (атрибут `cpd`) принимает значения, соответствующие
        равномерному распределению. Так что последние две строчки в данном примере можно было бы опустить.

    Факторы удобны для использования в вероятностных графических моделях тем, что на них определены некоторые операции,
    часто выполняемые в процессе логического вывода, в довольно общей форме, что позволяет использовать их как
    элементарные объекты для построения логических сетей.

    Также, факторы могут представлять условные вероятности. Рассмотрим пример из медицинской диагностики. Определим
    бинарную переменную с=(«болезнь», {«нет», «есть»}) и бинарную переменную
    t=(«тест», {«положительный», «отрицательный»}). Переменная с показывает, есть ли у пациента определенное
    заболевание. а переменная t – результат диагностического теста. Допустим, что априорная вероятность данного
    заболевания равна 1%. Сконструируем фактор С=(«P(c)», {c}, {0.99, 0.01}). Очевидно, что вероятность получения
    определенного результата теста зависит от того, есть ли данное заболевание у пациента, или нет, то есть мы
    предполагаем известной вероятность P(t|c). Допустим, что вероятность получения положительного результата теста
    равна 90%. Соответственно, вероятность ложноотрицательного результата равна 0,1. Вероятность получить отрицательный
    результат при отсутствии заболевания равна 0,8, а вероятность ложноположительного результата – 0,2.
    Создадим фактор, представляющий данную условную вероятность::

        c = Variable(name='C', terms=['no', 'yes'])
        t = Variable(name='T', terms=['pos', 'neg'])
        C = Factor(name='C', cons=[c])
        C.cpd = np.array([0.99, 0.01])
        T = Factor(name='T|C', cons=[t], cond=[c])
        T.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])

    Синтаксис:

        Создадим два фактора, представляющих распределение двух переменных: P(A) и P(B|A).
        Первая переменная имеет безусловное распределение, значение второй зависит от значения первой.

        Подготовка переменных (могут быть как дискретные, так и со связанным классификатором):

        >>> from pyinference.fuzzy import set as fuzzy_set
        >>> a = Variable(name='A', terms=['low', 'high'])
        >>> fs = fuzzy_set.Partition(peaks=[0.0, 0.5, 1.0])
        >>> b = Variable(name='B', terms=fs)

        Создание факторов:

        >>> A = Factor(name='A', cons=[a])
        >>> B = Factor(name='B|A', cons=[b], cond=[a])

    Поля класса:
        name (`str`): имя фактора

        cond (`list`): список условных переменных

        cons (`list`): список подусловных переменных

        vars (`list`): список всех переменных фактора (объединение предыдущих двух)

        shape (`tuple`): кортеж мощностей всех переменных фактора (сохраняя порядок атрибута `vars`).
            Соответствует форме массива `cpd`.

        cpd (:class:`numpy.array`): массив, хранящий распределение условной вероятности фактора.

    Именованные параметры:
        name (`str`): имя фактора

        cons (`list`): массив подусловных переменных

        cond (`list`): массив условных переменных

    Исключения:
        AttributeError: ошибка возникает, если массив подусловных переменных (cons) пуст
    """

    def __init__(self, name='', cond=None, cons=None):
        if len(cons) == 0:
            raise AttributeError
        self.name = name
        self.cond = sorted(cond) if cond else []
        self.cons = sorted(cons)
        self.vars = self.cond + self.cons
        self.shape = tuple([var.card for var in self.vars])

        self.cpd = ones(self.shape)
        self._normalize()

    def _normalize(self):
        n, m = len(self.cons), len(self.cond)
        s = self.cpd.sum(axis=tuple(range(m, n + m))).flatten()
        koef = array(self.shape)[m:].prod()
        if isinstance(s, float):
            s = array([s])
        s = repeat(s, koef)
        s[s == 0.0] = 1.0
        self.cpd = self.cpd.flatten() / s
        self.cpd = self.cpd.reshape(self.shape)

    def _map(self, other):
        res = []
        for i in xrange(len(other.vars)):
            for j in xrange(len(self.vars)):
                if self.vars[j].name == other.vars[i].name:
                    res.append(j)
                    break
        return res

    def marginal(self, var):
        """ Выполняет маргинализацию переменной из фактора.

        Маргинализация позволяет исключить переменную из области определения фактора, просуммировав соответствующие
        значения назначений маргинализируемого фактора. Маргинализация является основой алгоритма variable elimination:

        - F(A,B,C) - B = F(A,C)
        - F(A,C|B) - B = F(A,C)
        - F(A,B|C) - B = F(A|C)

        Может вызываться как метод (``m = f1.marginal(var)``) или как оператор "-" (``m = f1 - var``).

        Синтаксис:
            >>> import numpy as np
            >>> c = Variable(name='C', terms=['no', 'yes'])
            >>> t = Variable(name='T', terms=['pos', 'neg'])
            >>> C = Factor(name='C', cons=[c])
            >>> C.cpd = np.array([0.99, 0.01])
            >>> T = Factor(name='T|C', cons=[t], cond=[c])
            >>> T.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])
            >>> m = T - c
            >>> m.cpd
            array([ 1.1,  0.9])
            >>> len(m.vars)
            1

        Параметры:
            var (:class:`Variable` or `list`): маргинализуемая (исключаемая) переменная.
                Также в данный метод может передаваться список исключаемых переменных.

        Возвращает:
            Маргинализированный фактор

        Исключения:
            `TypeError`: ошибка возникает, когда второй операнд имеет неподдерживаемый тип.
        """
        ind = -1
        for i in xrange(len(self.vars)):
            if var.name == self.vars[i].name:
                ind = i
        if ind == -1:
            raise AttributeError
        cond1 = set(self.cond)
        cons1 = set(self.cons)
        cons2 = {var}
        cond = cond1 - cons2
        cons = cons1 - cons2
        res = Factor(name="Marginal", cons=sorted(list(cons)), cond=sorted(list(cond)))
        res.cpd = self.cpd.sum(axis=ind)
        return res

    def product(self, other):
        """ Реализует произведение факторов.

        Произведение факторов объединяет области определения двух факторов. Значением для каждого назначения является
        произведение соответствующих назначений множителей:

        - F(A) * F(B) = F(A,B)
        - F(A,|B) * F(B) = F(A,B)
        - F(A,B|C) * F(D|C) = F(A,B,D|C)
        - F(A,B|C) * F(D,C) = F(A,B,C,D)
        - None * F(A) = F(A)

        Может вызываться как метод (``p = f1.product(f2)``) или как оператор "*" (``p = f1 * f2``).

        .. note::
            Также первым операндом произведения может выступать None (``None * f1``). Тогда метод вернет второй операнд.
            Это сделано для удобства множественного произведения.

        Синтаксис:
            >>> import numpy as np
            >>> c = Variable(name='C', terms=['no', 'yes'])
            >>> t = Variable(name='T', terms=['pos', 'neg'])
            >>> C = Factor(name='C', cons=[c])
            >>> C.cpd = np.array([0.99, 0.01])
            >>> T = Factor(name='T|C', cons=[t], cond=[c])
            >>> T.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])
            >>> p = T * C
            >>> "%0.3f" % p.cpd[0,0]
            '0.198'
            >>> len(p.vars)
            2

        Параметры:
            other (:class:`Factor`): фактор-множитель.

        Возвращает:
            Фактор-произведение двух исходных

        Исключения:
            `AttributeError`: ошибка возникает, когда исключаемая переменная не входит в исходный фактор.

            `TypeError`: ошибка возникает, когда второй операнд имеет неподдерживаемый тип.
        """
        cond1 = set(self.cond)
        cons1 = set(self.cons)
        cond2 = set(other.cond)
        cons2 = set(other.cons)

        everything = cons1 | cons2 | cond1 | cond2
        cond = cond1 & cond2
        cons = everything - cond
        res = Factor(name="Product", cons=list(cons), cond=list(cond))

        flat = res.cpd.flatten()
        ass = _itershape(res.shape)
        map1 = res._map(self)
        map2 = res._map(other)

        for i in range(len(flat)):
            ass1, ass2 = [], []
            for j in map1:
                ass1.append(ass[i][j])
            for j in map2:
                ass2.append(ass[i][j])
            cpd1 = self.cpd[tuple(ass1)]
            cpd2 = other.cpd[tuple(ass2)]
            flat[i] = cpd1 * cpd2
        res.cpd = flat.reshape(res.shape)
        return res

    def divide(self, other):
        """ Реализует деление факторов.

        Деление факторов позволяет получить условное распределение из безусловного:

        F(A,B) / F(B) = F(A|B)

        Может вызываться как метод (``p = f1.divide(f2)``) или как оператор "/" (``p = f1 / f2``).

        Синтаксис:
            >>> import numpy as np
            >>> c = Variable(name='C', terms=['no', 'yes'])
            >>> t = Variable(name='T', terms=['pos', 'neg'])
            >>> C = Factor(name='C', cons=[c])
            >>> C.cpd = np.array([0.99, 0.01])
            >>> T = Factor(name='T|C', cons=[t], cond=[c])
            >>> T.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])
            >>> p = (C * T) / C
            >>> "%0.3f" % p.cpd[0,0]
            '0.200'
            >>> len(p.vars)
            2

        Параметры:
            other (:class:`Factor`): фактор-делитель.

        Возвращает:
            Фактор-частное двух исходных

        Исключения:
            NotImplementedError: ошибка возникает, когда:
                - делитель имеет условные переменные;
                - некоторые подусловные переменные делителя отсутствуют среди подусловных переменных делимого

            TypeError: ошибка возникает, когда второй операнд имеет неподдерживаемый тип.
        """
        cond1 = set(self.cond)
        cons1 = set(self.cons)
        cond2 = set(other.cond)
        cons2 = set(other.cons)

        if len(cond2) > 0:
            raise NotImplementedError
        if len(cons2 & cond1) > 0:
            raise NotImplementedError
        if len(cons2 & cons1) < len(cons2):
            raise NotImplementedError

        cond = cond1 | cons2
        cons = cons1 - cons2
        res = Factor(name="Conditional", cons=sorted(list(cons)), cond=sorted(list(cond)))

        flat = res.cpd.flatten()
        ass = _itershape(res.shape)
        map1 = res._map(self)
        map2 = res._map(other)

        for i in range(len(flat)):
            ass1, ass2 = [], []
            for j in map1:
                ass1.append(ass[i][j])
            for j in map2:
                ass2.append(ass[i][j])
            cpd1 = self.cpd[tuple(ass1)]
            cpd2 = other.cpd[tuple(ass2)]
            flat[i] = cpd1 / cpd2
        res.cpd = flat.reshape(res.shape)
        res._normalize()
        return res

    def __mul__(self, other):
        if other is None:
            return self
        elif not isinstance(other, Factor):
            raise TypeError
        return self.product(other)

    def __rmul__(self, other):
        if other is None:
            return self
        elif isinstance(other, Factor):
            return self.product(other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, Variable):
            return self.marginal(other)
        elif isinstance(other, list):
            res = self
            for var in other:
                res = res - var
            return res
        else:
            raise TypeError

    def __div__(self, other):
        if not isinstance(other, Factor):
            raise TypeError
        return self.divide(other)

    def __repr__(self):
        """ Подробное текстовое представление фактора.

        Возвращает:
            строку, содержащую слудующую информацию о факторе:
            - имя фактора;
            - имена подусловных и условных переменных, разделенных символом "|" (порядок внутри групп может
            не соблюдаться);
            - список всех переменных фактора в соответствующем порядке;
            - распределение условной вероятности по векторам значений переменных в соответственном порядке;
            - сумму вектора распределения (должна быть равна 1.0 для безусловных распределений).
        """
        res = ''
        res += self.name + ':\n'
        flat = self.cpd.flatten()
        ass = _itershape(self.shape)
        res += str(self.cons) + '|' + str(self.cond) + '\n'
        res += str(self.vars) + '\n'
        res += str(self.shape) + ', ' + str(self.cpd.shape) + '\n'
        for i in range(len(ass)):
            res += str(ass[i]) + '    ' + str(flat[i]) + '\n'
        res += 'Sum: ' + str(self.cpd.sum()) + '\n'
        return res
