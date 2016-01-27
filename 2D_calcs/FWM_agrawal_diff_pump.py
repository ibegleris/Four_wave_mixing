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
from scipy.integrate import  ode
from scipy.io import loadmat
from scipy.integrate import dblquad
import mkl
from scipy.optimize import fsolve
import sys
#import joblibn
plt.ioff()
mkl.set_num_threads(8)

from functions_2pumps import *
def main():
  ############### The constants
  n2 = 1.1225783990826979e-20 # nonlinear coefficient
  lamda_c = 1.5508e-6         # the central freequency that the betas are calculated around
  ###############



  num_steps = 10 # initial number of steps for the integrator
  d = 1e3 # propagation distance

  zmin =0 #starting point of the fibre
  zmax = d #end point of interest
  z = np.linspace(zmin,zmax,num_steps) #linearly spaced propagation vector
  dz = 1 * (z[1] - z[0])
  num_steps = (zmax - zmin) / dz +1


  ############## What beam is in what mode
  B = []
  B.append('lp01') #pump1
  B.append('lp11') #pump2
  B.append('lp01') #sigal
  B.append('lp11') #Idler
  #############

  w_vec = np.loadtxt('../loading_data/widths.dat')
  w_vec *=1e6
  overlap1,overlap2,zeroing = calc_overlaps(B,w_vec)
  print('calculated the overlaps going for the ode')
  #sys.exit()
  #P_vec1,P_vec2,  P_signal_vec, lamp1,lamp2,lams,lami = input_powers_wavelengths()


  lami = []


  ############ For the brag scattering inputs
  P_vec1 = dbm2w(np.array([18.0+3.3+10]))
  P_vec2 = dbm2w(np.array([13.8+3.3+10]))#dbm2w(np.array([-5]))*np.ones(len(P_vec2))+ dbm2w(10)
  P_signal_vec = dbm2w(np.array([-15.0+10]))#dbm2w(np.array([-7.5]))*np.ones(len(P_vec2)) - dbm2w(np.array([-42,-43,-45,-46,-45]))
  #P_signal_vec = dbm2w(P_signal_vec)

  lamp1 = np.array([1549.8]) * 1e-9
  lamp2 = np.array([1553.8e-9]) # guess
  lams =np.array([1548.8]) * 1e-9


  ###########

  ########### Find where the optimum freequency for pump2 is and plot the waves in vecor format for that freequency

  mat_lp = loadmat('../loading_data/coeffs.mat')
  lamp_min = fsolve(effective_phase_matching,lamp2,args = (n2,P_vec1,P_vec2,P_signal_vec,lamp1,lams,lami,dz,num_steps,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2))

  lamp2 = lamp_min#np.arange(1.546,1.558,0.0001)*1e-6##np.array([1552.6,1553.7,1553.9,1554.1,1554.2]) * 1e-9#
  AB_final = np.zeros([4,len(P_vec1),len(P_vec2),len(P_signal_vec),len(lamp1),len(lamp2),len(lams)],dtype='complex')
  Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])

  AB_final,lami,Dk = FWM(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,0,lams,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2)
  #plt.plot(lamp2*1e6,Dk[0,:,0])
  #plt.show()

  lam_arrows = np.array([lamp1[0],lamp2[0],lams[0],lami[-1::][0]])
  lam_arrows *=1e6
  powers_arrows = []
  for i in range(4):
      powers_arrows.append(w2dbm(np.abs(AB_final[i,0,0,0,0,0,0])**2))
  powers_arrows = np.asanyarray(powers_arrows)
  colours = ['red','green','blue','black']
  waves = ['pump1','pump2','signal','idler']
  fig = plt.figure()
  plt.ylim(-40,35)
  plt.xlim(1.548,1.555)
  plt.xlabel(r'$\lambda ( \mu m)$')
  plt.ylabel(r'$P(dBm)$')
  arrow = [1,2,3,4]
  for i in range(4):
      arrow[i] = plt.arrow(lam_arrows[i],-40,0,powers_arrows[i]+40,head_width=1e-4,head_length=1,width = 1e-12,length_includes_head=True,color=colours[i])
  plt.legend([arrow[0],arrow[1],arrow[2],arrow[3]], [waves[0]+','+B[0],waves[1]+','+B[1],waves[2]+','+B[2],waves[3]+','+B[3]],loc=0)
  plt.title('Linear phase matching:'+ r'$\lambda_p = $'+str(lamp_min[0]*1e6)+r'$\mu m$')
  #plt.savefig('plots/arrows_linear.png', bbox_inches='tight')
  plt.show()

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
  lamp2 = np.linspace(lamp_min - 0.001*lamp_min,lamp_min + 0.001*lamp_min,1024)
  AB_final = np.zeros([4,len(P_vec1),len(P_vec2),len(P_signal_vec),len(lamp1),len(lamp2),len(lams)],dtype='complex')
  Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])
  #sys.exit  # = FWM(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lams,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2)
  from joblib import Parallel, delayed
  A = Parallel(n_jobs=6)(delayed(FWM)(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2_,m,lams,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2) for m,lamp2_ in enumerate(lamp2)) 
  A = np.asanyarray(A)
  #AB_final = A[:,0]
  Dk = np.zeros([len(lamp2)])
  for i in range(len(lamp2)):
      for j in range(4):    
          AB_final[j,0,0,0,0,i,0] =A[i,0][j][0][0][0][0][i][0]
      Dk[i] = A[i,2][0][i][0]
  fig = plt.figure()
  plt.plot(lamp2*1e6,w2dbm(np.abs(AB_final[3,0,0,0,0,:,0])**2))
  plt.ylabel(r'$Power (dBm)$')
  plt.xlabel(r'$\lambda_{p2}(\mu m)$')
  plt.title('Effective phase match for varying pump2 wavelength')
  plt.show()

  fig = plt.figure()
  plt.plot(lamp2*1e6,Dk)
  plt.ylabel(r'$\Delta k(1/m)$')
  plt.xlabel(r'$\lambda_{p2}(\mu m)$')
  plt.show()
  Dk_vec,Dk_vec_nl = effective_phase_matching_general(lamp2,n2,P_vec1,P_vec2,P_signal_vec,lamp1,lams,lami,dz,num_steps,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2)
  fig =plt.figure()
  plt.plot(lamp2*1e6,Dk_vec[0,:,0],label = 'L')
  plt.plot(lamp2*1e6,Dk_vec_nl[0,:,0],label = 'NL')
  #plt.plot(lamp2*1e6,Dk_vec_nl[0,:,0] - Dk_vec[0,:,0],label = 'NL - L')
  plt.ylabel(r'$\Delta k(1/m)$')
  plt.xlabel(r'$\lambda_{p2}(\mu m)$')
  plt.legend()
  plt.show()

  fig =plt.figure()
  #plt.plot(lamp2*1e6,Dk_vec[0,:,0],label = 'L')
  #plt.plot(lamp2*1e6,Dk_vec_nl[0,:,0],label = 'NL')
  plt.plot(lamp2*1e6,Dk_vec_nl[0,:,0] - Dk_vec[0,:,0],label = 'NL - L')
  plt.ylabel(r'$(\Delta k_L - \Delta k_{NL} )(1/m)$')
  plt.xlabel(r'$\lambda_{p2}(\mu m)$')
  plt.legend()
  plt.show()
  return

# Had to overwrite the previous FWM function since the paralelised value in this case is the second pump.
del FWM
def FWM(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2_,m,lams,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2):
        for l, lamp1_ in enumerate(lamp1):
            for n,lams_ in enumerate(lams):

                 lam_vec = np.array([lamp1_,lamp2_,lams_,1])
                 omega = 2*pi*c/(lam_vec[:])
                 omega[3] = omega[0] + omega[1] -omega[2]
                 lami.append(2*pi*c/omega[3])
                 Dk_vec[l,m,n] = Dk_func(omega,lamda_c,B,zeroing,mat_lp)
                 Dk = Dk_vec[l,m,n]
                 dz = 1/(np.abs(Dk)*10)
                 #dz=
                 dz=0.1
                 #dz=0.01
                 if Dk == 0 or dz > 1000:# or dz >=1000 or dz>1:
                     dz =0.1
                 #print dz

                 for i,P1 in enumerate(P_vec1):
                     for j,P2 in enumerate(P_vec2):
                         for k, P_signal in enumerate(P_signal_vec):

                            AB0 = np.array([P1,P2, P_signal, 0], dtype='complex')
                            AB0[:] = AB0[:]**0.5
                            int_method = 'dopri5'
                            AB_final[:,i,j,k,l,m,n],outcome = \
                                            integrator(int_method,AB0,zmin,zmax,dz,omega,Dk,n2,overlap1,overlap2)
                            if outcome==False:
                                #dz =100
                                print('first try failed, trying adaptive steping...')
                                exits = 0
                                int_method = 'dop853'
                                while exits<=55 and outcome == False:
                                    AB_final[:,i,j,k,l,m,n],outcome = \
                                                   integrator(int_method,AB0,zmin,zmax,dz,omega,Dk,n2,overlap1,overlap2)
                                    exits +=1
                                    dz *= 0.5
                                    print 'failed, trying step size:', dz,'...'
                                if outcome == False:
                                    sys.exit('All integrations failed')
        return AB_final,lami,Dk_vec


if __name__ == "__main__":
  main()

