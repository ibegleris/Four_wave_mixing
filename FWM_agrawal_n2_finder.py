# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:35:27 2015

@author: Ioannis Begleris
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from scipy.constants import pi,c
from scipy.integrate import ode
from scipy.io import loadmat
from scipy.optimize import fsolve
import sys
from numpy.linalg import norm
from peakdetect import peakdetect
plt.ioff()

def dbm_to_w(dBm):
    """This function converts a power given in dBm to a power given in W."""
    return 1e-3*10**((dBm)/10.)
def w_to_dbm(W):
    """This function converts a power given in W to a power given in dBm."""
    return 10.*np.log10(W)+30

def integrand(i, A, zz,omega,Dk,n2,over1):
    """Calculates the integrand of the 3 coupled ode's according to
        Fiber-Based Optical Parametric Amplifiers
        and Their Applications
        """
    try:
        n2 = n2[0]
    except:
        pass
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


def system(zz,AB,omega,Dk,n2,over1):
    """
        The system of differentail equations to be solved. Converts the floats
        to coplex and is used by scipy.integrate.ode
    """

    try: #This is put here because ode & odeint take their x,y values the other way round
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
        dABdz[i] = integrand(i,A,zz,omega[i],Dk,n2,over1)
    return dABdz.view(np.float64)


def integrator(int_method,AB0,zmin,zmax,dz,AB_final,i,j,omega,Dk,n2,over1):
    """
    Performs the integration using the scipy.integrate.ode class. the int_method
    determines the method to be used.
    """

    r = ode(system).set_integrator(int_method)
    r.f_params =(omega,Dk,n2,over1,)
    r.set_initial_value(AB0.view(np.float64),np.float64(zmin))
    count = 1
    num_steps = (zmax - zmin)/dz +1
    while count < num_steps and r.successful():
            r.integrate(r.t+dz)
            count+=1
    if r.successful():
        AB_final[:,i,j] = r.y.view(np.complex128)
    return AB_final[:,i,j], r.successful()

def FWM(n2,P_vec,dz,num_steps,zmin,zmax,lamp,lams,lami,beta2,P_signal_vec,over1):
    """
    Simmulates the FWM of the system by calling the integration function.
    If the first time fails then it halves the step until found. If not exits
    with an error.
    """
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
            #print (omega[1]**2+omega[2]**2-2*omega[0]**2)
            #print '...'
            print (omega[1]**2+omega[2]**2-2*omega[0]**2)
            int_method = 'dop853' #DP method of integration
            AB_final[:,i,j],outcome = \
                            integrator(int_method,AB0,zmin,zmax,dz,AB_final,i,j,omega,Dk,n2,over1)
            if not(outcome):
                print('first try failed, trying adaptive steping...')
                exits = 0
                int_method = 'dop853'
                while exits<=55:
                    AB_final[:,i,j],outcome = \
                                   integrator(int_method,AB0,zmin,zmax,dz,AB_final,i,j,omega,Dk,n2,over1)
                    exits +=1
                    dz *= 0.5
                    print 'failed, trying step size:', dz,'...'
            if not(outcome):
                sys.exit('integration failed completely')
    return AB_final

def searcher(n2,P_vec,dz,num_steps,AB_measured,zmin,zmax,lamp,lams,lami,beta2,P_signal_vec,over1):

    AB_final = FWM(n2,P_vec,dz,num_steps,zmin,zmax,lamp,lams,lami,beta2,P_signal_vec,over1)

    return  norm(np.abs(AB_final[2])**2 - dbm_to_w(AB_measured[2]),2)


def inputs(filename):
    """
    Takes in the experimental values measured from the experiment that is
    simmulated to find n2.
    """
    mat  = loadmat('data/LP11_FWM_data.mat')
    #mat = loadmat(filename)
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
    return AB_measured,P_signal_vec,P_vec,lamp,lams,lami


def plotter(norms,n2_foundp,n2_found, lams,lami,AB_final, AB_measured):
    """
    Self explanatory
    """
    D_p_sim,  D_s_sim,  D_i_sim =  unpack(AB_final)

    D_p,  D_s,  D_i =  AB_measured[0,0,:] ,AB_measured[1,0,:], AB_measured[2,0,:]
    D_p += 10
    fig = plt.figure()
    plt.plot(n2_foundp,norms)
    plt.xlabel(r'$n_2$')
    plt.ylabel(r'$I_{error}$')
    plt.title('$n_2$:'+str(n2_found[0][0])+', err:'+str(n2_found[1]['fvec']))
    plt.savefig('n2.png')
    plt.close(fig)


    fig = plt.figure()
    plt.plot(lams,w_to_dbm(D_p_sim),'x',label='sim')
    plt.plot(lams,D_p,'o',label = 'meas')
    plt.plot(lams,np.abs(D_p - w_to_dbm(D_p_sim)),label='diff')
    plt.xlabel(r'$\lambda (m)$')
    plt.title('Pump')
    plt.ylabel(r'$P(dbm)$')
    plt.legend(loc=(1.01,0.805))
    plt.savefig('Pump.png',bbox_inches='tight')
    plt.close(fig)



    fig = plt.figure()
    plt.plot(lams,w_to_dbm(D_s_sim),'x',label='sim' )
    plt.plot(lams,np.abs(D_s - w_to_dbm(D_s_sim)),label='diff')
    plt.plot(lams,D_s,'o',label = 'meas')
    plt.xlabel(r'$\lambda (m)$')
    plt.title('Signal')
    plt.ylabel(r'$P(dbm)$')
    plt.legend(loc=(1.01,0.805))
    plt.savefig('Signal.png',bbox_inches='tight')
    plt.close(fig)


    fig = plt.figure()

    plt.plot(lami,w_to_dbm(D_i_sim),'x',label='sim')
    plt.plot(lami,D_i,'o',label = 'meas')
    plt.plot(lami,np.abs(D_i - w_to_dbm(D_i_sim)),label='diff')
    plt.xlabel(r'$\lambda (m)$')
    plt.title('Idler')
    plt.ylabel(r'$P(dbm)$')
    plt.legend(loc=(1.01,0.805))
    plt.savefig('Idler.png',bbox_inches='tight')
    plt.close(fig)
    return
def energy_conservation(P_signal_vec,P_vec,AB_final):
    D_p_sim,  D_s_sim,  D_i_sim = unpack(AB_final)
    DeltaE = P_signal_vec + P_vec*np.ones(len(P_signal_vec)) - \
            D_i_sim - D_p_sim - D_s_sim
    print 'Difference in watts (inf norm)',np.linalg.norm(DeltaE,np.inf)
    return



def unpack(AB_final):
    """
    unpacks the powers for the pump signal and idler.
    """
    D_p_sim = np.abs(AB_final[0,0,:])**2
    D_s_sim = np.abs(AB_final[1,0,:])**2
    D_i_sim = np.abs(AB_final[2,0,:])**2
    return  D_p_sim,  D_s_sim,  D_i_sim







def main():


    n2 = 3e-20 # nonlinear coefficient (guess)
    num_steps = 10 # initial number of steps for the integrator, if fails the stepis refined
    d = 1e3 # Propagation distance of the waves
    mode='lp11' # mode that is simmulated, only LP01 LP11 with no cross mode interaction(only one mode launched)
    lamda_c = 1.5508e-6 #The central freequency at which the dispersion is calculated
    zmin =0 #starting point of the fibre
    zmax = d #end point of interest
    z = np.linspace(zmin,zmax,num_steps) #linearly spaced propagation vector

    dz = 1 * (z[1] - z[0])
    del z
    num_steps = (zmax - zmin) / dz +1
    if mode == 'lp01':
        filename = 'data/LP01_FWM_data.mat'
        def over1():
            return (161e-12)**(-1)
        DD = 19.9*1e-6
    else:
        filename = 'data/LP11_FWM_data.mat'
        def over1():
            return (170e-12)**(-1)
        DD = 22*1e-6
    AB_measured,P_signal_vec,P_vec,lamp,lams,lami = inputs(filename)
    print AB_measured
    beta2 = -lamda_c**2*DD/(2*pi*c)
    #print n2_found
    #sys.exit()
    #bound=((1.5e-20,2e-20),)
    #opts = {'maxiter' : None,'disp' : True, 'gtol' : 1e-30, 'norm' : 1e-30,'eps' : 1.4901161193847656e-30}  # default value.
    #n2_found = minimize(searcher,n2,method = 'TNC',args=(P_vec,dz,num_steps,AB_measured),bounds = bound,tol = 1e-30)
    #fmin_slsqp
    #print searcher(n2_found,P_vec,dz,num_steps,AB_measured)
    #n2_found = fmin(searcher,n2,args=(P_vec,dz,num_steps,AB_measured),full_output=1)
    #n2_found = fmin_slsqp(searcher,n2,bounds = bound,args=(P_vec,dz,num_steps,AB_measured))

    #n2_found = minimize(searcher,n2,args=(P_vec,dz,num_steps,AB_measured))
    #print searcher(n2_found,P_vec,dz,num_steps,AB_measured)



    n2_foundp = np.arange(1e-22,3e-20,1e-21)
    norms = np.zeros(len(n2_foundp))

    for i in range(len(norms)):
        norms[i] = searcher(n2_foundp[i],P_vec,dz,num_steps,AB_measured,zmin,zmax,lamp,lams,lami,beta2,P_signal_vec,over1)
    #print 'found' , n2_found[0]
    #print 'idler norm:', searcher(n2_found[0],P_vec,dz,num_steps,AB_measured)




    n2_found =fsolve(searcher,n2,args=(P_vec,dz,num_steps,AB_measured,zmin,zmax,lamp,lams,lami,beta2,P_signal_vec,over1),full_output=1,xtol=1e-33)
    print 'Found: n2 = ', n2_found[0][0]
    print 'L2 norm', n2_found[1]['fvec']
    AB_final = FWM(n2_found[0],P_vec,dz,num_steps,zmin,zmax,lamp,lams,lami,beta2,P_signal_vec,over1)

    plotter(norms,n2_foundp,n2_found, lams,lami,AB_final, AB_measured)
    energy_conservation(P_signal_vec,P_vec,AB_final)
    return 0

if __name__ == '__main__':
    main()
