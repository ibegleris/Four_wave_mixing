
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:35:27 2015

@author: Ioannis Begleris
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from scipy.constants import pi,c
from scipy.integrate import  ode
from scipy.io import loadmat
import sys
from peakdetect import peakdetect
plt.ioff()

def dbm_to_w(dBm):
    """This function converts a power given in dBm to a power given in W."""
    return 1e-3*10**((dBm)/10.)
def w_to_dbm(W):
    """This function converts a power given in W to a power given in dBm."""
    return 10.*np.log10(W)+30

def integrand(i, A, zz,omega,Dk,a,n2):
        if i==0:
            temp = A[0]*(np.abs(A[0])**2 + 2*np.abs(A[1])**2 + 2*np.abs(A[2])**2) + \
                    2*np.conj(A[0])*A[1]*A[2]*np.exp(1j*Dk*zz)
        elif i==1:
            temp = A[1]*(np.abs(A[1])**2 + 2*np.abs(A[2])**2 + 2*np.abs(A[0])**2) + \
                    np.conj(A[2])*(A[0]**2)*np.exp(-1j*Dk*zz)

        elif i ==2:
            temp = A[2]*(np.abs(A[2])**2 + 2*np.abs(A[1])**2 + 2*np.abs(A[0])**2) + \
                    np.conj(A[1])*(A[0]**2)*np.exp(-1j*Dk*zz) 
            

        gama = over1()*n2*omega/c
        return 1j*gama *temp


def system(zz,AB,omega,Dk,a,n2):
    try:
        temp = AB[0]
    except:
        temp = AB
        AB = np.copy(zz)        
        zz = np.copy(temp)
        del temp
        pass
    dABdz=np.zeros(3,dtype='complex')
    A0 = AB[0] + AB[1]*1j
    A1 = AB[2] + AB[3]*1j
    A2 = AB[4] + AB[5]*1j
    A = np.array([A0, A1, A2])
    
    for i in range(3):
        dABdz[i] = integrand(i,A,zz,omega[i],Dk,a,n2)
    return dABdz.view(np.float64) 


def integrator(int_method,AB0,zmin,zmax,dz,AB_final,i,j,omega,Dk,a,n2):
    r = ode(system).set_integrator(int_method)
    r.f_params =(omega,Dk,a,n2,)
    r.set_initial_value(AB0.view(np.float64),np.float64(zmin))
    count = 1
    num_steps = (zmax - zmin)/dz +1
    while count < num_steps and r.successful():
            r.integrate(r.t+dz)
            count+=1
    if r.successful():
        AB_final[:,i,j] = r.y.view(np.complex128)      
    return AB_final[:,i,j], r.successful()



n2 = 1.1225783990826979e-20 # nonlinear coefficient
num_steps = 10 # initial number of steps for the integrator
d = 1e3 # propagation distance
mode='lp11' # mode that is simmulated
lamda_c = 1.5508e-6
zmin =0 #starting point of the fibre
zmax = d #end point of interest

z = np.linspace(zmin,zmax,num_steps) #linearly spaced propagation vector
dz = 1 * (z[1] - z[0])
num_steps = (zmax - zmin) / dz +1
a= 0
if mode == 'lp01':
    def over1():
        return (161e-12)**(-1)
    DD = 19.9*1e-6
else:
    def over1():
        return (170e-12)**(-1)
    DD = 22*1e-6

beta2 = -lamda_c**2*DD/(2*pi*c) 



mat  = loadmat('../loading_data/LP11_FWM_data.mat')
lams_vec_exp= mat['lam_vals']
D = mat['D']
del mat
lamp = np.zeros(len(D[0,:])); lams = np.zeros(len(D[0,:])); lami = np.zeros(len(D[0,:])) 

D_p = np.zeros(len(D[0,:])); D_s = np.zeros(len(D[0,:])); D_i = np.zeros(len(D[0,:])) 
for i in range(len(D[0,:])):
    _max, _min = peakdetect(D[:,i], lams_vec_exp[:,i], 50)
    max_ = np.asanyarray(_max)
    max_ = max_[np.argsort(max_[:,1])]
    lamp[i], D_p[i] = max_[-1::,0][0],max_[-1::,1][0]
    lami[i], D_i[i] = max_[-3::3,0][0],max_[-3::3,1][0]
    lams[i], D_s[i] = max_[-2::2,0][0],max_[-2::2,1][0]

D_p = D_p[0:-3:]
D_s = D_s[0:-3:]
D_i = D_i[0:-3:]

lamp = lamp[0:-3:]
lams = lams[0:-3:]
lami = lami[0:-3:]
P_vec = np.arange(22.7,23.7,2)
P_vec +=10
P_vec = dbm_to_w(P_vec)

P_signal_vec = dbm_to_w(D_s) - dbm_to_w(D_i)



lamp = np.copy(lamp)*1e-9
lams = np.copy(lams)*1e-9
lami = np.copy(lami)*1e-9

AB_measured = np.zeros([3,len(P_vec),len(lams)])
AB_measured[0,0,:] = D_p
AB_measured[1,0,:] = D_s
AB_measured[2,0,:] = D_i


def FWM(n2,P_vec,dz,num_steps,a):
    AB_final = np.zeros([3,len(P_vec),len(lams)],dtype='complex')
    for i,Po in enumerate(P_vec):
        P = Po  
        for j,lams_ in enumerate(lams):
            P_signal = P_signal_vec[j]
            
            AB0 = np.array([P,P_signal, 0], dtype='complex')
            AB0[:] = AB0[:]**0.5

            lami_ = lami[j]
            lamp_ = lamp[j]
            lam_vec = np.array([lamp_,lams_,lami_])

            omega = 2*pi*c/(lam_vec[:])

            omega[2] = 2*omega[0]-omega[1]
            lami[j] = 2*pi*c/omega[2]
            
            Dk = 0.5*beta2*(omega[1]**2+omega[2]**2-2*omega[0]**2)
            int_method = 'dop853' 
            AB_final[:,i,j],outcome = \
                            integrator(int_method,AB0,zmin,zmax,dz,AB_final,i,j,omega,Dk,a,n2)
            if outcome==False:
                print('first try failed, trying adaptive steping...')            
                exits = 0
                int_method = 'dop853'  
                while exits<=55:
                    AB_final[:,i,j],outcome = \
                                   integrator(int_method,AB0,zmin,zmax,dz,AB_final,i,j,omega,Dk,a,n2) 
                    exits +=1           
                    dz *= 0.5
                    print 'failed, trying step size:', dz,'...'
                if outcome == False:
                    sys.exit('All integrations failed')
    return AB_final


AB_final = FWM(n2,P_vec,dz,num_steps,a)



fig = plt.figure()
for i, pp in enumerate(w_to_dbm(P_vec)):
    D_p_sim = np.abs(AB_final[0,i,:])**2
    plt.plot(lams,w_to_dbm(D_p_sim),'x',label='pp')
plt.xlabel(r'$\lambda (m)$')
plt.title('Pump')
plt.ylabel(r'$P(dbm)$')
plt.legend(loc=(1.01,0.805))
plt.savefig('Pump.png',bbox_inches='tight')
plt.close(fig)



fig = plt.figure()
for i, pp in enumerate(w_to_dbm(P_vec)):
    D_s_sim = np.abs(AB_final[1,i,:])**2
    plt.plot(lams,w_to_dbm(D_s_sim),'x',label='diff')
plt.xlabel(r'$\lambda (m)$')
plt.title('Signal')
plt.ylabel(r'$P(dbm)$')
plt.legend(loc=(1.01,0.805))
plt.savefig('Signal.png',bbox_inches='tight')
plt.close(fig)


fig = plt.figure()
for i, pp in enumerate(w_to_dbm(P_vec)):
    D_i_sim = np.abs(AB_final[2,i,:])**2
    plt.plot(lami,w_to_dbm(D_i_sim),'x',label='sim')
plt.xlabel(r'$\lambda (m)$')
plt.title('Idler')
plt.ylabel(r'$P(dbm)$')
plt.legend(loc=(1.01,0.805))
plt.savefig('Idler.png',bbox_inches='tight')
plt.close(fig)




DeltaE = P_signal_vec + P_vec*np.ones(len(P_signal_vec)) - \
        D_i_sim - D_p_sim - D_s_sim
print 'Infinity norm of the energy difference', np.linalg.norm(DeltaE,np.inf)
