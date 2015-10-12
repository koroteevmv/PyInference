# coding=utf-8
from pyinference.inference import Factor, Variable, Net
import numpy as np


three=Variable(name="3", terms=["low","mean","high"])
five=Variable(name="5", terms=["low","mean","high"])
seven=Variable(name="7", terms=["low","mean","high"])
nine=Variable(name="9", terms=["low","mean","high"])
ten=Variable(name="10", terms=["low","mean","high"])


Nine=Factor(name="F9", cons=[nine])

Three=Factor(name="F3", cons=[three],cond=[nine])
Three.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))

Five=Factor(name="F5", cons=[five],cond=[three, nine])
Five.cpd=np.array([0.8,0.1,0.1,0.5,0.5,0,0,1,0,0.3,0.5,0.2,0.1,0.6,0.3,0.3,0.4,0.3,0,1,0,0.2,0.4,0.4,0.1,0.1,0.8]).reshape((3,3,3))

Seven=Factor(name="F7", cond=[five],cons=[seven])
Seven.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))

Ten=Factor(name="F10", cond=[seven],cons=[ten])
Ten.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))


BN=Net(name="123",nodes=[Nine, Three, Five, Seven, Ten])

q=BN.query(query=[nine], evidence=[seven])

print q