# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:39:10 2016

@author: john
"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.io import loadmat, savemat
import matplotlib.pylab as plt
import numpy as np
import sys

def w2dbm(W):
    """This function converts a power given in W to a power given in dBm.
       Inputs::
           W(float): power in units of W
       Returns::
           Power in units of dBm(float)
    """
    return 10.*np.log10(W)+30
#AB_final = A[:,0]

def plot(AB_final,lams,lamp2):
    X,Y = np.meshgrid(lams,lamp2)
    Z = AB_final
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_xlabel(r'$\lambda_s(\mu m)$')
    ax.set_ylabel(r'$\lambda_p(\mu m)$')
    ax.set_zlabel(r'$\P_{idler}(dBm)$')

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    fig = plt.figure()
    axi = plt.contourf(X,Y,Z, cmap=cm.jet)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.xlabel(r'$\lambda_s(\mu m)$')
    plt.ylabel(r'$\lambda_p(\mu m)$')
    fig.colorbar(axi, shrink=0.5, aspect=5)
    plt.show()
    return 0

dic = loadmat('Export_data/brag.mat')
A = dic['A']
lamp2_start_brag = 1.553
lamp2_end_brag = 1.554
lams_start_brag = 1.549
lams_end_brag = 1.553
lamp2_brag = np.linspace(lamp2_start_brag, lamp2_end_brag,256)
lams_brag = np.linspace(lams_start_brag, lams_end_brag,256)
A = np.asanyarray(A)
del dic
AB_final = np.zeros([256,256])
for i in range(len(lams_brag)):
    for j in range(len(lamp2_brag)):
        AB_final[i,j] = w2dbm(np.abs(A[i][3][j][i])**2)
dict_save = {}
dict_save['lamp2_brag'] = lamp2_brag
dict_save['lams_brag'] = lams_brag
dict_save['P_idler_brag'] = AB_final
plot(AB_final,lams_brag,lamp2_brag)
sys.exit()

dic = loadmat('Export_data/phase_conj.mat')
A = dic['A']
del dic
lamp2_start_phase_conj = 1.550
lamp2_end_phase_conj = 1.554
lams_start_phase_conj = 1.549
lams_end_phase_conj = 1.553
A = np.asanyarray(A)

lamp2_phase_conj = np.linspace(lamp2_start_phase_conj, lamp2_end_phase_conj,256)
lams_phase_conj = np.linspace(lams_start_phase_conj, lams_end_phase_conj,256)
A = np.asanyarray(A)

AB_final = np.zeros([256,256])
for i in range(len(lams_phase_conj)):
    for j in range(len(lamp2_phase_conj)):
        AB_final[i,j] = w2dbm(np.abs(A[i][3][j][i])**2)

dict_save['lamp2_phase_conj'] = lamp2_phase_conj
dict_save['lams_phase_conj'] = lams_phase_conj
dict_save['P_idler_phase_conj'] = AB_final

savemat('idler_power.mat', dict_save)

plot(AB_final,lams_phase_conj,lamp2_phase_conj)

