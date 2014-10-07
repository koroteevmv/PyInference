#-------------------------------------------------------------------------------
# Name:
# Purpose:
#
# Author:      sejros
#
# Created:     15.04.2014
# Copyright:   (c) sejros 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import fuzzycalc

def main():
    import pylab as py
    names = ['1', '2', '3']
    A = fuzzycalc.set.GaussianClassifier(names=names, edge=True, cross=1)
    A.plot()
    py.show()

if __name__ == '__main__':
    main()
