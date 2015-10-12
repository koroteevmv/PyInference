# coding=utf-8
from pyinference.inference import Factor, Variable, Net
import numpy as np

posetiteli=Variable(name=" posetiteli ", terms=["low","mean","high"])
prosmotri=Variable(name=" prosmotri ", terms=["low","mean","high"])
vremia=Variable(name=" vremia ", terms=["low","mean","high"])
tochka_vhoda=Variable(name=" tochka_vhoda ", terms=["low","mean","high"])
konversia=Variable(name=" konversia ", terms=["low","mean","high"])
vozvrashaemost=Variable(name=" vozvrashaemost ", terms=["low","mean","high"])
otkazi=Variable(name=" otkazi ", terms=["low","mean","high"])

Tochka=Factor(name="dot", cons=[ tochka_vhoda])

Posetiteli=Factor(name="pos", cons=[ posetiteli],cond=[ tochka_vhoda])
Posetiteli.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))

Konversia =Factor(name="kon", cons=[ konversia],cond=[ vremia, prosmotri])
Konversia.cpd=np.array([0.9,0.08,0.02,0.3,0.4,0.3,0.05,0.25,0.7,0.3,0.5,0.2,0.05,0.75,0.2,0.15,0.65,0.2,0.05,0.45,0.5,0.04,0.26,0.7,0.01,0.1,0.89]).reshape((3,3,3))

Prosmotri=Factor(name="prosm", cond=[ posetiteli],cons=[ prosmotri])
Prosmotri.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))

Vremia=Factor(name="time", cond=[ prosmotri],cons=[ vremia])
Vremia.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))

Vozvrashaemost =Factor(name="vozv", cond=[ konversia],cons=[ vozvrashaemost])
Vozvrashaemost.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))

Otkazi =Factor(name="otk", cond=[ konversia],cons=[ otkazi])
Otkazi.cpd=np.array([1,0,0,0,1,0,0,0,1]).reshape((3,3))

BN=Net(name="123",nodes=[ Tochka, Posetiteli,
Prosmotri, Vremia, Konversia, Vozvrashaemost, Otkazi])

q=BN.query(query=[tochka_vhoda], evidence=[ vozvrashaemost])

print q
