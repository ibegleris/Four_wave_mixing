# -*- coding: utf-8 -*-
"""
This code, given athe inverse group velocity of the LP01 and LP11 modes find the optimum
wavelengths at which the idler is most amplified. (only works for specific cases due to the nonexistant
bet_0 in my problem ie when the pumps are in different modes)
@author: Ioannis Begleris
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from scipy.constants import pi,c
from scipy.io import loadmat
from scipy.optimize import fsolve
import sys
from accelerate import mkl
from joblib import Parallel, delayed
import pandas as pan
plt.ioff()

def plotter(lams,lami,AB,name):
    fig = plt.figure(figsize=(20.0, 10.0))
    ax1 = fig.add_subplot(111)
    ax1.plot(lami*1e6,AB)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    
    ax2 = ax1.twiny()
    ax2.plot(lams*1e6,AB,alpha=0)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel(r'$Power (dBm)$')
    #ax1.set_ylim(-60,0)
    ax1.set_xlabel(r'$\lambda_{i}(\mu m)$')
    ax2.set_xlabel(r'$\lambda_{s}(\mu m)$')
    plt.savefig(name+'.png',bbox_inches='tight')
    plt.close(fig)
    return 0
def plotter_deturing(lam,AB,lam_meas,conv_meas,name):
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.plot(lam*1e9,AB)
    plt.plot(lam_meas,conv_meas,'o')
    plt.xlabel(r'$\left|\lambda_p - \lambda_i \right| (nm)$')
    plt.ylabel('Conv_eff (dB)')
    plt.savefig(name+'.png',bbox_inches='tight')
    plt.close(fig)
    return 0
def energy_conservation(P_vec1,P_vec2,P_signal_vec,AB_final):
    E_init = np.ones(len(AB_final[0,0,0,0,0,0,:]))
    E_final = np.zeros(len(E_init))
    E_init *= (P_vec1+P_vec2+P_signal_vec)
    
    for i in range(4):
        E_final += np.abs(AB_final[i,0,0,0,0,0,:])**2
    return np.linalg.norm(E_init - E_final,np.inf)

def read_csv(w):
    which = ['one','two','three','four','five','six','seven']
    A = pan.read_csv('convers/conversion.csv')
    A = A.fillna(0)
    B = A.as_matrix()
    meas = conv_eff_meas(B, w)
    meas.clean()
    return meas


class conv_eff_meas:
    def __init__(self,B,w):
        self.lam = B[:,3*w]
        self.bs = B[:,3*w+1]
        self.pc = B[:,3*w+2]
    def clean(self):
        """
        cleans all the zero's out of the data set from the nans by pandas
        """
        self.lam = self.lam[np.nonzero(self.lam)[0]]
        self.bs = self.bs[np.nonzero(self.bs)[0]]
        self.pc = self.pc[np.nonzero(self.pc)[0]]


from functions_2pumps import *
mkl.set_num_threads(8)
def main(w,idl,corrr):
    ############################The constants&variables########################################
    n2 = 2.5e-20 #Silica 1.1225783990826979e-20 # nonlinear coefficient
    lamda_c = 1.5508e-6         # the central freequency that the betas are calculated around

    seeds  = 100
    num_eval = 1000
    num_steps = 1000 # initial number of steps for the integrator
    d = 1e3 # propagation distance
    monte_carlo_average = True
    print "monte_carlo_average" , monte_carlo_average
    if monte_carlo_average:
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
    #w = 0 # which pump2 wavelength?
    meas = read_csv(w)

    if monte_carlo_average:
        FWM_func = FWM_monte
    else:
        FWM_func = FWM
    # Loading the conversion effiency data



    #What beam is in what mode
    B = []
    B.append('lp01') #pump1
    B.append('lp11') #pump2
    B.append('lp01') #sigal
    B.append('lp11') #Idler
    LP01_cor = 0.6152567345983897 #[dBm]
    LP11_cor = -1.77386603355#[dBm]
    #The power and wavelength inputs
    P_vec1 = dbm2w(np.array([30.5 + LP01_cor]))
    P_vec2 = dbm2w(np.array([30.5 + LP11_cor - corrr]))
    P_signal_vec = dbm2w(np.array([30.5-25 +LP01_cor]))
    lamp1 = np.array([1549]) * 1e-9
    lamp2 = np.array([lam_meas[w]])*1e-9
    lams =np.array([1548.8]) * 1e-9  # guess
    lam_variation = np.linspace(1.549,1.5525,512)*1e-6
    ##########################################################################################
    if idl == 'bs':
        P_vec1,P_signal_vec = P_signal_vec,P_vec1

    ###############################Find the overlaps####################################
    w_vec = np.loadtxt('../loading_data/widths.dat')
    w_vec *=1e6
    overlap1,overlap2,zeroing = calc_overlaps(B,w_vec)
    print('calculated the overlaps going for the ode')
    ####################################################################################


    ###########Find where the optimum freequency for signal is and plot the waves in vecor format for that freequency########################################################
    mat_lp = loadmat('../loading_data/coeffs.mat')
    lami = []
    lams_min = fsolve(effective_phase_matching,lams,args = (n2,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lami,dz,num_steps,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2))
    ##########################################################################################################################################################################


    #################################################Do the FWM for the optimum wavelength#################################################################
    lams = lams_min
    AB_final = np.zeros([4,len(P_vec1),len(P_vec2),len(P_signal_vec),len(lamp1),len(lamp2),len(lams)],dtype='complex')
    Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])
    AB_final,lami,Dk = FWM(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams[0],0,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor,seeds,num_eval,idl)
    ########################################################################################################################################################


    ###############################################################Plot the arrows##########################################################################
    lam_arrows = np.array([lamp1[0],lamp2[0],lams[0],lami[-1::][0]])
    lam_arrows *=1e6
    powers_arrows = []
    for i in range(4):
        powers_arrows.append(w2dbm(np.abs(AB_final[i,0,0,0,0,0,0])**2))
    powers_arrows = np.asanyarray(powers_arrows)
    colours = ['red','green','blue','black']
    waves = ['pump1','pump2','signal','idler']
    fig = plt.figure(figsize=(20.0, 10.0))
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    #fig.set_size_inches(100,100)
    plt.ylim(-40,35)
    plt.xlim(min(lam_arrows)- min(lam_arrows)*0.001,max(lam_arrows)+max(lam_arrows)*0.001)
    plt.xlabel(r'$\lambda ( \mu m)$')
    plt.ylabel(r'$P(dBm)$')
    arrow = [1,2,3,4]
    for i in range(4):
        arrow[i] = plt.arrow(lam_arrows[i],-40,0,powers_arrows[i]+40,head_width=1e-4,head_length=1,width = 1e-12,length_includes_head=True,color=colours[i])
    plt.legend([arrow[0],arrow[1],arrow[2],arrow[3]], [waves[0]+','+B[0],waves[1]+','+B[1],waves[2]+','+B[2],waves[3]+','+B[3]],loc=0)
    plt.title('Linear phase matching:'+ r'$\lambda_s = $'+str(lams_min[0]*1e6)+r'$\mu m$')
    plt.savefig('arrows.png',bbox_inches='tight')
    plt.close(fig)
    #########################################################################################################################################################


    #############################################################################Do the calculations for a wide grid###################################################################################
    #if idl=='bs':
    #    lamp1 = lam_variation
    #else:
    #    lams = lam_variation
    ##sys.exit()
    lams = lam_variation
    AB_final = np.zeros([4,len(P_vec1),len(P_vec2),len(P_signal_vec),len(lamp1),len(lamp2),len(lams)],dtype='complex')
    Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])
    #A = Parallel(n_jobs=6)(delayed(FWM)(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams_,n,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor) for n,lams_ in enumerate(lams)) 

    #if idl == 'pc':
    A = Parallel(n_jobs=6)(delayed(FWM_func)(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams_,n,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor,seeds,num_eval,idl) for n,lams_ in enumerate(lams)) 
    A = np.asanyarray(A)
    #AB_final = A[:,0]
    lami = np.zeros(len(lam_variation))
    Dk = np.zeros([len(lamp2)])
    for i in range(len(lams)):
        lami[i] = A[:,1][i][1]
        for j in range(4):    
            AB_final[j,0,0,0,0,0,i] =A[i,0][j][0][0][0][0][0][i]
    print "Energy conservation check. Worst case senario:", energy_conservation(P_vec1,P_vec2,P_signal_vec,AB_final)
    #sys.exit()
    Powers = np.zeros(np.shape(AB_final))
    Powers = w2dbm(np.abs(AB_final)**2)
    Powers -= np.max(Powers)
    #AB_final = dbm2w(w2dbm(AB_final) - np.max(w2dbm(AB_final)))
    #maxim = 0#np.max(w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2))
    conv_eff = np.zeros(len(lams))
    if idl == 'pc':
        #conv_eff[:] = w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2) - w2dbm(np.abs(AB_final[2,0,0,0,0,0,:])**2)
        conv_eff[:] = Powers[3,0,0,0,0,0,:] - Powers[2,0,0,0,0,0,:]
        
    else:
        #conv_eff[:] = w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2) - w2dbm(np.abs(AB_final[0,0,0,0,0,0,:])**2)
        conv_eff[:] = Powers[3,0,0,0,0,0,:] - Powers[0,0,0,0,0,0,:]
        #plotter(lams,lami,Powers[3,0,0,0,0,0,:],'idler_large')
        #plotter(lams,lami,Powers[2,0,0,0,0,0,:],'signal_large')
        #plotter(lams,lami,Powers[1,0,0,0,0,0,:],'pump2_large')
        #plotter(lams,lami,Powers[0,0,0,0,0,0,:],'pump1_large')

    plotter(lams,lami,Powers[3,0,0,0,0,0,:],'idler_large')
    plotter(lams,lami,Powers[2,0,0,0,0,0,:],'signal_large')
    plotter(lams,lami,Powers[1,0,0,0,0,0,:],'pump2_large')
    plotter(lams,lami,Powers[0,0,0,0,0,0,:],'pump1_large')
    #plotter(lams,lami,w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2),'idler_large')
    #plotter(lami,lams,w2dbm(np.abs(AB_final[2,0,0,0,0,0,:])**2),'signal_large')

    #plotter(lams,lami,w2dbm(np.abs(AB_final[1,0,0,0,0,0,:])**2),'pump2_large')
    #plotter(lams,lami,w2dbm(np.abs(AB_final[0,0,0,0,0,0,:])**2),'pump1_large')
   
    """
    #else:
        A = Parallel(n_jobs=6)(delayed(FWM_func)(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lams,lamp1_,l,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor,seeds,num_eval) for l,lamp1_ in enumerate(lamp1)) 
        A = np.asanyarray(A)
        #AB_final = A[:,0]
        lami = np.zeros(len(lam_variation))
        Dk = np.zeros([len(lamp2)])
        for i in range(len(lams)):
            lami[i] = A[:,1][i][1]
            for j in range(4):    
                AB_final[j,0,0,0,i,0,0] =A[i,0][j][0][0][0][i][0][0]
        AB_final = dbm2w(w2dbm(AB_final) - np.max(w2dbm(AB_final)))
        maxim = 0#np.max(w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2))
        conv_eff = np.zeros(len(lams))
        conv_eff[:] = w2dbm(np.abs(AB_final[3,0,0,0,:,0,0])**2) - w2dbm(np.abs(AB_final[2,0,0,0,:,0,0])**2)


        plotter(lams,lami,w2dbm(np.abs(AB_final[3,0,0,0,:,0,0])**2),'idler_large')
        plotter(lams,lami,w2dbm(np.abs(AB_final[0,0,0,0,:,0,0])**2),'power_large')
    """




    plotter(lams,lami,conv_eff,'conv_large')
    #plotter_deturing(, , 'conv_deturing_large')
    plotter_deturing(np.abs(lami-lamp2),conv_eff,meas.lam,eval('meas.'+idl),'det2'+str(-1*corrr)+'/conversion_effieicny_det_'+idl+'_'+str(w))
    #plotter(lams,lami[::-1],w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2),'idler_large')
    #plotter(lams,lami[::-1],w2dbm(conv_eff),'conv_large')
    #plotter_deturing(lami[::-1]-lams, w2dbm(conv_eff), 'conv_deturing_large')
    ###################################################################################################################################################################################################
    return AB_final
print "starting"
for i in range(7):
    print main(i,'bs',6)
    print "phase conj"
    print main(i,'pc',6)
    print main(i,'bs',0)
    print "phase conj"
    print main(i,'pc',0)
"""
###############################################################Do the calculations for the magnified amount in the middle##########################################################################
lams = np.linspace(1.5525,1.5545,256)*1e-6
AB_final = np.zeros([4,len(P_vec1),len(P_vec2),len(P_signal_vec),len(lamp1),len(lamp2),len(lams)],dtype='complex')
Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])

A = Parallel(n_jobs=6)(delayed(FWM_func)(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams_,n,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor,seeds,num_eval) for n,lams_ in enumerate(lams)) 
lami = []
Dk_vec,Dk_vec_nl,lami = effective_phase_matching_general(lams,n2,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lami,dz,num_steps,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2)
A = np.asanyarray(A)
lami = np.zeros(len(lams))
Dk = np.zeros([len(lamp2)])
for i in range(len(lams)):
    lami[i] = A[:,1][i][1]
    for j in range(4):    
        AB_final[j,0,0,0,0,0,i] =A[i,0][j][0][0][0][0][0][i]

conv_eff = np.zeros(len(lams),dtype=np.complex)
conv_eff[:] = np.abs(AB_final[3,0,0,0,0,0,:])**2/np.abs(AB_final[2,0,0,0,0,0,:])**2


plotter(lams,lami,w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2),'idler_zoom')
plotter(lams,lami,w2dbm(conv_eff),'conv_zoom')
plotter_deturing(np.abs(lami-lamp2), w2dbm(conv_eff), 'conv_deturing_zoom')
######################################################################################################################################################################################################
#    return

if __name__ == "__main__":
    main()
"""    


"""
lam_vec = np.array([lamp1[0],lamp2[0],lams[0],1])
omega = 2*pi*c/(lam_vec[:])
omega[3] = omega[0] + omega[1] -omega[2]

xpm0 = np.loadtxt('0.txt')
xpm1 = np.loadtxt('1.txt')
xpm2 = np.loadtxt('2.txt')
xpm3 = np.loadtxt('3.txt')
z_plot = np.loadtxt('z.txt')
dknl0 = xpm2 + xpm3 - xpm1
dknl1 = xpm2+xpm1 -xpm0
dknl2 = xpm1+xpm0 -xpm3
dknl3 = xpm2+xpm1 -xpm2


fig = plt.figure()
plt.plot(z_plot,dknl0)
plt.xlabel(r'$z(m)$')
plt.ylabel(r'$\Delta k_{NL}$')
plt.title('pump1 XPM_SPM')
plt.show()


fig = plt.figure()
plt.plot(z_plot,dknl1)

plt.title('pump2 XPM_SPM')
plt.xlabel(r'$z(m)$')
plt.ylabel(r'$\Delta k_{NL}$')
plt.show()


fig = plt.figure()
plt.plot(z_plot,dknl2)

plt.title('signal XPM_SPM')
plt.xlabel(r'$z(m)$')
plt.ylabel(r'$\Delta k_{NL}$')
plt.show()


fig = plt.figure()
plt.plot(z_plot,dknl3)
plt.title('idler XPM_SPM')
plt.xlabel(r'$z(m)$')
plt.ylabel(r'$\Delta k_{NL}$')
plt.show()


sys.exit()
"""
"""
dzs = []
dz =0.1
for i in range(6):
    dzs.append(dz)
    dz *=0.5
del dz
lamp2 = np.asanyarray([lamp_min + 0.1*lamp_min])
converge = []
for dz  in dzs:
    print dz
    AB_final,lami,Dk = FWM(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lams,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor)
    converge.append(np.abs(AB_final[3,0,0,0,0,0,0])**2)
plt.plot(dzs,converge)
plt.show()
sys.exit()
"""
