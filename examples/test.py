# coding=utf-8

import numpy as np
from pyinference.inference import Variable, Factor, Net

BN = Net()

a = Variable(name='a', terms=[0, 1])
A = Factor(name='A', cons=[a])
A.cpd = np.array([0.5, 0.5])
BN.add_node(A)

b = Variable(name='b', terms=[0, 1])
B = Factor(name='B', cons=[b])
B.cpd = np.array([0.5, 0.5])
BN.add_node(B)

c = Variable(name='c', terms=[0, 1])
C = Factor(name='C', cons=[c], cond=[a, b])
C.cpd = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).reshape(C.shape)
BN.add_node(C)

q1 = BN.query(query=[a, b], evidence=[c])
print q1

print q1 / B