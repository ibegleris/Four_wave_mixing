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
plt.ioff()
mkl.set_num_threads(8)

from functions_2pumps import *
############################The constants&variables########################################
n2 = 1.1225783990826979e-20 # nonlinear coefficient
lamda_c = 1.5508e-6         # the central freequency that the betas are calculated around

num_steps = 10 # initial number of steps for the integrator
d = 1e3 # propagation distance

zmin =0 #starting point of the fibre
zmax = d #end point of interest
z = np.linspace(zmin,zmax,num_steps) #linearly spaced propagation vector
dz = 1 * (z[1] - z[0])
num_steps = (zmax - zmin) / dz +1


#What beam is in what mode
B = []
B.append('lp01') #pump1
B.append('lp11') #pump2
B.append('lp01') #sigal
B.append('lp11') #Idler


#The power and wavelength inputs
P_vec1 = dbm2w(np.array([21.3+10]))
P_vec2 = dbm2w(np.array([21.3+10]))
P_signal_vec = dbm2w(np.array([-3.7+10]))
lamp1 = np.array([1549]) * 1e-9
lamp2 = np.array([1553.6])*1e-9
lams =np.array([1548.8]) * 1e-9  # guess
##########################################################################################


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
AB_final,lami,Dk = FWM(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams[0],0,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2)
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
#########################################################################################################################################################


#############################################################################Do the calculations for a wide grid###################################################################################
lams = np.linspace(1.545,1.553,512)*1e-6
AB_final = np.zeros([4,len(P_vec1),len(P_vec2),len(P_signal_vec),len(lamp1),len(lamp2),len(lams)],dtype='complex')
Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])
A = Parallel(n_jobs=6)(delayed(FWM)(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams_,n,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2) for n,lams_ in enumerate(lams)) 
A = np.asanyarray(A)
#AB_final = A[:,0]
Dk = np.zeros([len(lamp2)])
for i in range(len(lams)):
    for j in range(4):    
        AB_final[j,0,0,0,0,0,i] =A[i,0][j][0][0][0][0][0][i]
    #Dk[i] = A[i,2][0][0][i]
fig = plt.figure(figsize=(20.0, 10.0))
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.plot(lams*1e6,w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2))

plt.ylabel(r'$Power (dBm)$')
plt.xlabel(r'$\lambda_{p2}(\mu m)$')
plt.title('Idler power for varying signal wavelength')
plt.savefig('large.png',bbox_inches='tight')
###################################################################################################################################################################################################


###############################################################Do the calculations for the magnified amount in the middle##########################################################################
lams = np.linspace(1.5525,1.5545,256)*1e-6
AB_final = np.zeros([4,len(P_vec1),len(P_vec2),len(P_signal_vec),len(lamp1),len(lamp2),len(lams)],dtype='complex')
Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])


A = Parallel(n_jobs=6)(delayed(FWM)(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams_,n,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2) for n,lams_ in enumerate(lams)) 
lami = []
Dk_vec,Dk_vec_nl,lami = effective_phase_matching_general(lams,n2,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lami,dz,num_steps,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2)
A = np.asanyarray(A)

Dk = np.zeros([len(lamp2)])
for i in range(len(lams)):
    for j in range(4):    
        AB_final[j,0,0,0,0,0,i] =A[i,0][j][0][0][0][0][0][i]
    #Dk[i] = A[i,2][0][0][i]
fig = plt.figure(figsize=(20.0, 10.0))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()

ax1.plot(lams*1e6,w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2))
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.ylabel(r'$Power (dBm)$')
ax1.set_xlabel(r'$\lambda_{p2}(\mu m)$')
new_tick_locations = np.array([.2, .5, .9])
#plt.title('Idler power for varying signal wavelength')
ax2.plot(lami, np.zeros(len(k)))
ax2.set_xlabel('lami')
plt.show()
plt.savefig('zoom.png',bbox_inches='tight')
#plt.show()
#import peakdetect
#aaa = peakdetect.peakdetect(w2dbm(np.abs(AB_final[3,0,0,0,0,0,:])**2),lams,30)
#aaa[1][0][1] - aaa[0][0][1]
#print aaa
######################################################################################################################################################################################################


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
    AB_final,lami,Dk = FWM(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lams,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2)
    converge.append(np.abs(AB_final[3,0,0,0,0,0,0])**2)
plt.plot(dzs,converge)
plt.show()
sys.exit()
"""
