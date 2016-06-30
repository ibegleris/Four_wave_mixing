# -*- coding: utf-8 -*-
"""
This code, given athe inverse group velocity of the LP01 and LP11 modes find the optimum
wavelengths at which the idler is most amplified. (only works for specific cases due to the nonexistant
bet_0 in my problem ie when the pumps are in different modes)
@author: Ioannis Begleris
"""
from __future__ import division
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pylab as plt
from scipy.constants import pi,c
from scipy.io import loadmat, savemat
from scipy.optimize import fsolve
import sys
from accelerate import mkl
from joblib import Parallel, delayed
import pandas as pan
from functions_2pumps import *
#plt.ioff()

def plotter(lams,lami,AB,name):
    fig = plt.figure(figsize=(20.0, 10.0))
    ax1 = fig.add_subplot(111)
    ax1.plot(lami*1e6,AB)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    
    ax2 = ax1.twiny()
    ax2.plot(lams*1e6,AB,alpha=0)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel(r'$Power (dBm)$',fontsize=18)
    #ax1.set_ylim(-60,0)

    ax1.set_xlabel(r'$\lambda_{i}(\mu m)$',fontsize=18)
    ax2.set_xlabel(r'$\lambda_{s}(\mu m)$',fontsize=18)
    plt.savefig(name+'.png',bbox_inches='tight')
    plt.close(fig)
    plt.show()
    return 0
def plotter_deturing(lam,AB,lam_meas,conv_meas,name,idl):
    lam = [i*1e9 for i in lam]
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.plot(lam,AB,label ='Simulated')
    plt.plot(lam_meas,conv_meas,'-',label ='Experimental')
    if idl == 'bs':
        plt.xlabel(r'$\left|\lambda_{BS} - \lambda_{p2}\right| (nm)$',fontsize=18)
    else:
        plt.xlabel(r'$\left|\lambda_{PC} - \lambda_{p2}\right| (nm)$',fontsize=18)
    plt.ylabel('CE (dB)',fontsize=18)
    plt.ylim(-60,5)
    plt.grid()
    plt.legend()
    plt.savefig(name+'.png',bbox_inches='tight')
    plt.savefig(name+'.eps',bbox_inches='tight')
    #plt.show()
    plt.close(fig)
    
    return 0
def energy_conservation(P_vec1,P_vec2,P_signal_vec,AB_final):
    E_init = np.ones(len(AB_final[0,0,0,0,0,0,:]))
    E_final = np.zeros(len(E_init))
    E_init *= (P_vec1+P_vec2+P_signal_vec)
    
    for i in range(4):
        E_final += np.abs(AB_final[i,0,0,0,0,0,:])**2
    return np.linalg.norm(E_init - E_final,np.inf)

class conv_eff_meas(object):
    def __init__(self,lam_p2):

        dat = pan.read_csv('data/conversion.csv')
        lam_meas = dat[str(lam_p2)]
        lam_meas = lam_meas.as_matrix()
        lam_meas = lam_meas[~np.isnan(lam_meas)]

        bs_meas = dat[str(lam_p2)+'_BS']
        bs_meas = bs_meas.as_matrix()
        bs_meas = bs_meas[~np.isnan(bs_meas)]

        pc_meas = dat[str(lam_p2)+'_PC']
        pc_meas = pc_meas.as_matrix()
        pc_meas = pc_meas[~np.isnan(pc_meas)]
        self.lam = lam_meas
        self.bs = bs_meas
        self.pc = pc_meas




mkl.set_num_threads(8)
def main(w,idl,corrr,lp1,lp2,extra,num,calcoverlaps=True):
    ############################The constants&variables########################################
    n2 = 2.5e-20 #Silica 1.1225783990826979e-20 # nonlinear coefficient
    lamda_c = 1.5508e-6        # the central freequency that the betas are calculated around
    extra = int(not(extra))
    seeds  = 100
    num_eval = 200
    num_steps = 10000 # initial number of steps for the integrator
    d = 1000 # propagation distance
    noise = False
    print "Noise floor" , noise
    if noise:
        noise_floor = -60+30.5-1.6897557023157006 #[dbm] The lowest that the detector is seeing
        noise_floor = dbm2w(noise_floor)
    else:
        noise_floor = 0
    zmin =0 #starting point of the fibre
    zmax = d #end point of interest
    z = np.linspace(zmin,zmax,num_steps) #linearly spaced propagation vector
    dz = 1 * (z[1] - z[0])
    num_steps = (zmax - zmin) / dz +1
    lam_meas = np.loadtxt('convers/pump_wavelengths.csv')
    #meas = read_csv(w)
    print lp2
    meas = conv_eff_meas(lp2)
    if noise:
        FWM_func = FWM_monte
    else:
        FWM_func = FWM



    #What beam is in what mode
    B = []
    B.append('LP01') #pump1
    B.append('LP11') #pump2
    B.append('LP01') #signal
    B.append('LP11') #Idler
    LP01_cor = -3 #0.6152567345983897 #[dBm]
    LP11_cor = -3 #-1.77386603355#[dBm]
    P_vec1 = dbm2w(np.array([30.5 + LP01_cor]))
    P_vec2 = dbm2w(np.array([30.5 + LP11_cor]))
    P_signal_vec = dbm2w(30.5 + LP01_cor - 24) * np.ones(num)

    lamp1 = np.array([lp1]) * 1e-9
    lamp2 = np.array([lp2]) * 1e-9
    lams =np.array(1548.8)  * 1e-9  # guess
    lami = []
    lam_variation =np.linspace(lamp1[0],lamp1[0]+2.25e-9,num)
    print(lam_variation[0],lam_variation[::-1][0])
    ##########################################################################################
    if idl == 'bs':
        P_vec1,P_signal_vec = P_signal_vec,P_vec1

    ###############################Find the overlaps####################################
    w_vec = np.loadtxt('../loading_data/widths.dat')
    w_vec *= 1e6
    if calcoverlaps ==True:
        overlap1,overlap2,zeroing = calc_overlaps(B,w_vec)
        #sys.exit()
        print('calculated the overlaps going for the ode')
        D = {}
        D['overlap1'] = overlap1
        D['overlap2'] = overlap2
        D['zeroing'] = zeroing
        savemat('data/overlaps.mat', D)
    else:
        D = loadmat('data/overlaps.mat')
        overlap1 = D['overlap1'][0]
        overlap2 = D['overlap2'][0]
        zeroing = D['zeroing'][0]

    ####################################################################################

    mat_lp = loadmat('../loading_data/coeffs.mat')
    #############################################################################Do the calculations for a wide grid##################################################################################
    lams = lam_variation
    AB_final = np.zeros([4,len(lams)],dtype='complex128')
    Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])
    
    for n,lams_ in enumerate(lams):
        print(lams_)
        AB_final[:,n],lami = FWM(n2,AB_final,P_vec1,P_vec2,P_signal_vec,lamp1[0],lams_,n,lamp2[0],lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor,seeds,num_eval,idl)
    #print "Energy conservation check. Worst case senario:", energy_conservation(P_vec1,P_vec2,P_signal_vec,AB_final)
    Powers = np.zeros(np.shape(AB_final))
    Powers = w2dbm(np.abs(AB_final)**2)
    Powers -= np.max(Powers)

    conv_eff = np.zeros(len(lams))
    if idl == 'pc':
        conv_eff[:] = Powers[3,:] - Powers[2,:]
        np.savetxt('data/signal_power_pc.txt', Powers[2,:])
    elif idl == 'bs':
        Pow = np.loadtxt('data/signal_power_pc.txt')
        conv_eff[:] = Powers[3,:] - Pow

    #plotter(lams,np.asanyarray(lami),Powers[3,:],'idler_large')
    #plotter(lams,np.asanyarray(lami),Powers[2,:],'signal_large')
    #plotter(lams,np.asanyarray(lami),Powers[1,:],'pump2_large')
    #plotter(lams,np.asanyarray(lami),Powers[0,:],'pump1_large')

    
    plotter_deturing(np.abs(lami-lamp2),conv_eff,meas.lam,eval('meas.'+idl),'figures'+'/conversion_effieicny_det_'+idl+'_'+str(lp2),idl)
    D = {}
    D['lamp1'] = lamp1
    D['lamp2'] = lamp2
    D['lam_variation'] = lam_variation
    D['lami'] = lami
    D['lam_detur'] = np.abs(lami-lamp2)
    D['conv_eff'] = conv_eff
    D['lam_detur_exp'] = meas.lam
    D['conv_eff_det'] = eval('meas.'+idl)
    savemat('figures'+'/exported_data'+idl+str(lp2)+'.mat',D)


    return Powers, lami,lamp2

if __name__ == '__main__':
    for i,lp2 in enumerate((1553.4,)):
        AB_final, lami,lamp2 = main(i,'pc',0,1549,lp2,0,128,True)
        AB_finall, lamii,lamp22 = main(i,'bs',0,1549,lp2,0,128,True)
    #for i in range(9):
    #   break
    #   main(i,'bs',0,1.549,1.552,0)
    #   main(i,'pc',0,1.549,1.552,0)