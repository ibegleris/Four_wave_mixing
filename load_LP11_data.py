# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:38:55 2015

@author: john
"""

from __future__ import division
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import peakdetect as peak
mat  = loadmat('LP11_FWM_data.mat')
lams_vec= mat['lam_vals']
D = mat['D']
num = 0
#for i in range(num+1):
#    plt.plot(lams_vec[:,i],D[:,i],'o')
#plt.xlabel(r'$\lambda (\mu m)$')
#plt.ylabel(r'$P(dbm)$')
#plt.show()


from math import pi
import pylab

    


_max, _min = peak.peakdetect(D[:,0], lams_vec[:,0], 50)
max_ = np.asanyarray(_max)
max_ = max_[np.argsort(max_[:,1])]
lamp, D_p = max_[-1::,0],max_[-1::,1]

lams, D_s = max_[-2::2,0],max_[-2::2,1]
lami, D_i = max_[-3::3,0],max_[-3::3,1]
xm = [p[0] for p in _max]
ym = [p[1] for p in _max]
xn = [p[0] for p in _min]
yn = [p[1] for p in _min]

plot = pylab.plot(x, y)
pylab.hold(True)
pylab.plot(xm, ym, 'r+')
pylab.plot(xn, yn, 'g+')


pylab.show()