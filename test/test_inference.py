# coding=utf-8

import unittest
import numpy as np

from pyinference.inference.factor import Factor
from pyinference.inference.net import Net
from pyinference.inference.variable import Variable
from pyinference.fuzzy.set import Partition


class TestVariable(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_equals_dict(self):
        a = Variable(name='A', terms=['low', 'high'])
        a.value = 'low'
        self.assertAlmostEqual(a.equals('low'), 1.0)
        self.assertAlmostEqual(a.equals('high'), 0.0)

    def test_equals_classifier(self):
        fs = Partition(peaks=[0.0, 0.5, 1.0])
        a = Variable(name='A', terms=fs)
        a.value = 0.5
        self.assertAlmostEqual(a.equals(0.5), 1.0)
        self.assertAlmostEqual(a.equals('1'), 1.0)
        self.assertAlmostEqual(a.equals('0'), 0.0)
        a.value = '1'
        self.assertAlmostEqual(a.equals('1'), 1.0)
        self.assertAlmostEqual(a.equals(0.25), 0.5)


class TestFactor(unittest.TestCase):
    def setUp(self):
        a = Variable(name='A', terms=['low', 'high'])
        fs = Partition(peaks=[0.0, 0.5, 1.0])
        b = Variable(name='B', terms=fs)

        self.A = Factor(name='A', cons=[a])
        self.B = Factor(name='B|A', cons=[b], cond=[a])

        self.c = Variable(name='C', terms=['no', 'yes'])
        self.t = Variable(name='T', terms=['pos', 'neg'])

        self.C = Factor(name='C', cons=[self.c])
        self.C.cpd = np.array([0.99, 0.01])
        # TODO cpd assignment
        self.T = Factor(name='T|C', cons=[self.t], cond=[self.c])
        self.T.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])

    def tearDown(self):
        pass

    def test__init__(self):
        self.assertAlmostEqual(self.A.shape, (2,))
        self.assertAlmostEqual(self.B.shape, (2, 3))

        self.assertAlmostEqual(self.A.cpd[0], 0.5)
        self.assertAlmostEqual(self.B.cpd[0, 0], 1/3.0)

    def test_map(self):
        a = Variable(name='A', terms=[0, 1])
        b = Variable(name='B', terms=[0, 1])
        c = Variable(name='C', terms=[0, 1])
        p1 = Factor(name='P(A,B,C)', cons=[a, b, c])
        p2 = Factor(name='P(A,B)', cons=[a, b])
        p3 = Factor(name='P(B,C)', cons=[b, c])
        map1 = p1._map(p2)
        map2 = p1._map(p3)
        self.assertListEqual(map1, [0, 1])
        self.assertListEqual(map2, [1, 2])

    def test_marginal(self):
        m = self.T - self.c
        self.assertAlmostEqual(1.1, m.cpd[0])
        self.assertAlmostEqual(0.9, m.cpd[1])
        self.assertEqual(1, len(m.vars))

    def test_product(self):
        p = self.C * self.T
        self.assertAlmostEqual(0.198, p.cpd[0, 0])
        self.assertAlmostEqual(0.001, p.cpd[1, 1])
        self.assertEqual(2, len(p.vars))

    def test_division(self):
        p = (self.C * self.T) / self.C
        self.assertAlmostEqual(0.2, p.cpd[0, 0])
        self.assertAlmostEqual(0.8, p.cpd[0, 1])
        self.assertEqual(2, len(p.vars))


class TestNet(unittest.TestCase):

    def setUp(self):
        self.c = Variable(name='C', terms=['no', 'yes'])
        self.t = Variable(name='T', terms=['pos', 'neg'])

        self.C = Factor(name='C', cons=[self.c])
        self.C.cpd = np.array([0.99, 0.01])
        self.T = Factor(name='T|C', cons=[self.t], cond=[self.c])
        self.T.cpd = np.array([[0.2, 0.8], [0.9, 0.1]])

        self.bn = Net(name='sample_net')

    def tearDown(self):
        pass

    def test_add_node(self):
        self.assertRaises(AttributeError, lambda : self.bn.add_node(self.T))
        self.bn.add_node(self.C)
        self.bn.add_node(self.T)

    def test_joint(self):
        bn = Net(name='Cancer', nodes=[self.C, self.T])
        j = bn.joint()
        self.assertTupleEqual((2,2), j.shape)
        self.assertEqual(2, len(j.vars))
        self.assertEqual('0.198', "%0.3f" % j.cpd[0,0])
        self.assertEqual('0.001', "%0.3f" % j.cpd[1,1])

    def test_query(self):
        bn = Net(name='Cancer', nodes=[self.C, self.T])
        q = bn.query(query=[self.c], evidence=[self.t])
        self.assertTupleEqual((2,2), q.shape)
        self.assertEqual(2, len(q.vars))
        self.assertEqual('0.957', "%0.3f" % q.cpd[0,0])
        self.assertEqual('0.001', "%0.3f" % q.cpd[1,1])


if __name__ == '__main__':
    unittest.main()