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
from joblib import Parallel, delayed
from accelerate import numba
jit  = numba.jit
autojit = numba.autojit

def dbm2w(dBm):
    """This function converts a power given in dBm to a power given in W.
       Inputs::
           dBm(float): power in units of dBm
       Returns::
           Power in units of W (float)
    """
    return 1e-3*10**((dBm)/10.)


def w2dbm(W):
    """This function converts a power given in W to a power given in dBm.
       Inputs::
           W(float): power in units of W
       Returns::
           Power in units of dBm(float)
    """
    return 10.*np.log10(W)+30

@jit
def xpm_spm(i,A,overlap1):
    """Calculates the XPM-SPM term in the FWM equations(Ref Agrawal 4th edition)
       Inputs::
           i(int): The wave that is looked at (0-p1,1-p2,2-s,3-i)
           A(complex vector shape[4]): Amplitude of ith wave
           overlap1(complex shape[4,4]: The first overlap integral

       Local::
           k(int)
           sum(complex)
       Returns(complex)::
           The XPM-SPM contribution to the amplitude change
    """
    sums = 0
   
    for k in range(4):
        if k != i:
            sums +=overlap1[i,k]*np.abs(A[k])**2
    sums *=2
    sums +=overlap1[i,i]*np.abs(A[i])**2
    
    return sums



def integrand(i, A, zz,omega,Dk,n2,overlap1,overlap2):
    """
    Calculates and returns the integrand on the ith ODE from Agrawal. Firstly the 
    FWM term is calculated then the XPM_SPM term is added upon.
     Inputs::
           i(int): The wave that is looked at (0-p1,1-p2,2-s,3-i)
           A(complex vector shape[4]): Amplitude of ith wave
           overlap1(complex shape[4,4]: The first overlap integral
           overlap1(complex shape[4,4]: The second(FWM) overlap integral
           n2(float): nonlinear coefficient
           omega(float): the angular freequency of ith wave
           Dk(float): the linear phase matching(delta beta)
           zz(float): Length of the fibre that the algorithm has propagated to.
       Local::
           i_vec(array shape[4,3]): global array to give symmetry of the integrand
           temp(float): temp value used for convienience
           gama(float): NOT the gama specified in Agrawal (due to overlaps)
       Returns(complex)::
           The integrand of the ith wave
    """
    i_vec = np.array([[1,2,3],[0,2,3],[3,0,1],[2,0,1]])
    ii = i_vec[i,:]
    if i//2 == 0:
        temp = 2*overlap2[i] * np.conj(A[ii[0]]) * A[ii[1]] * A[ii[2]] * np.exp(1j*Dk*zz)
    else:
        temp = 2*overlap2[i] * np.conj(A[ii[0]]) * A[ii[1]] * A[ii[2]] * np.exp(-1j*Dk*zz)
    temp +=xpm_spm(i,A,overlap1)*A[i]
    gama = n2*omega/c

    return 1j*gama *temp


def system(zz,AB,omega,Dk,n2,overlap1,overlap2):
    """
    The main use of this function is to bypass that most ode solving algorithms
    in scipy are problematic when dealing with complex numbers.
    Hence it takes an size 8 array and breaks it in to complex numbers before 
    calling the integrand. Afterwards the shape 8 array is reconstructed for 
    the ode solving algorithm.
    Inputs::
           zz(float): Length of the fibre that the algorithm has propagated to.
           AB(real vector shape[8]): Amplitude of ith wave in float view
           overlap1(complex shape[4,4]: The first overlap integral
           overlap1(complex shape[4]: The second(FWM) overlap integral
           n2(float): nonlinear coefficient
           omega(float): the angular freequency of ith wave
           Dk(float): the linear phase matching(delta beta)
       Local::
           i_vec(array shape[4,3]): global array to give symmetry of the integrand
           temp(float): temp value used for convienience
           gama(float): NOT the gama specified in Agrawal (due to overlaps)
       Returns(complex)::
           The XPM-SPM contribution to the amplitude change
    """
    try:
        temp = AB[0]
    except:
        temp = AB
        AB = np.copy(zz)
        zz = np.copy(temp)
        del temp
        pass
    dABdz=np.zeros(4,dtype='complex128')
    A0 = AB[0] + AB[1]*1j
    A1 = AB[2] + AB[3]*1j
    A2 = AB[4] + AB[5]*1j
    A3 = AB[6] + AB[7]*1j
    A = np.array([A0, A1, A2,A3])

    for i in range(4):
        dABdz[i] = integrand(i,A,zz,omega[i],Dk,n2,overlap1,overlap2)
    return dABdz.view(np.float64)


def integrator(int_method,AB0,zmin,zmax,dz,omega,Dk,n2,overlap1,overlap2):
    """
    Is used to setup and, via adaptinve steps, make sure that the integration is 
    successfull.
    Inputs::
           int_method: Length of the fibre that the algorithm has propagated to.
           AB0(complex vector shape[4]): Initial amplitude of waves in complex view
           overlap1(real array shape[4,4]: The first overlap integral
           overlap1(real vector shape[4]: The second(FWM) overlap integral
           zmin(float):Start point of the fibre
           zmax(float):Ending of the fibre
           dz(float): step size
           n2(float): nonlinear coefficient
           omega(float): the angular freequency of ith wave
           Dk(float): the linear phase matching(delta beta)
       Local::
           r(object): Integrator object
           num_steps(float): The number of steps needed to be taken
           count(int): Counts the number of steps taken
       Returns(complex)::
           AB_final : the final large array with all the values at the end
           r.successful(): whether the integration is sucessfull
    """
    r = ode(system).set_integrator(int_method)
    r.f_params =(omega,Dk,n2,overlap1,overlap2,)
    r.set_initial_value(AB0.view(np.float64),np.float64(zmin))
    count = 1
    num_steps = (zmax - zmin)/dz +1
    AB_final = np.zeros(4)
    while count <= num_steps and r.successful() and r.t+dz<1000.:
            r.integrate(r.t+dz)
            count+=1
    if r.t != 1000 and r.successful():
        #print 'catch'
        if r.t>=1000.0:
            print 'Oversteped by:', r.t - 1000.0
            sys.exit()
        else:
             count = 1
             #print 'taking step of', dz
             dz = (zmax - r.t)
             r.integrate(r.t+dz)
             
    if r.successful() and r.t == 1000:
      AB_final = r.y.view(np.complex128)
      return AB_final, r.successful()
    else:
      sys.exit()


def calc_overlaps(B,w_vec):
  B_func = [0, 0, 0 ,0]
  for i,b in enumerate(B):
      if b == 'LP01':
          B_func[i] = field0
      elif b == 'LP11':
          B_func[i] = field1
  overlap1 = np.zeros([4,4])
  for i in range(4):
      for j in range(4):
          overlap1[i,j] = over1(i,j,B,B_func,w_vec)
  i_vec = np.array([[1,2,3],[0,2,3],[3,0,1],[2,0,1]])
  overlap2 = np.zeros(4)
  zeroing = 0
  for i in range(4):
      ii = i_vec[i,:]
      overlap2[i] = over2(i,ii[0],ii[1],ii[2],B,B_func,w_vec)
      if overlap2[i] == 0:
          zeroing +=1
  overlap1 *=1e12
  overlap2 *=1e12
  return overlap1,overlap2,zeroing

@jit
def field0(y,x,w_vec):
    w = w_vec[0]
    return np.exp(-(x**2+y**2)/w**2)

@jit
def field1(y,x,w_vec):
    w = w_vec[1]
    return (2*2**0.5*x/w)*np.exp(-(x**2+y**2)/w**2)


def over1(i,j,B,B_func,w_vec):
    """
    Calculates the first overlap integral. If it is found that it is with itself then the
    inverse effective area is returned otherwise the integrals are calculated. For the mode calculations
    the hermit-gaussian approximation is taken.
    Also the calculation is done in terms of microns^2 and is transformed in to m^2 in calc_overlaps
    Inputs::
        i,j (int,int): Integer on what whave the overlap is calculated for
        B(str vec shape[4]): Holding the mode for each wave. (lp01 or lp11)
        B_func( function vec shape[4]) : Points to what function is used to calculate each mode(field0 or field1)
        w_vec(float vec shape[2]) : The width of the lp01 or the lp11 modes. (calculated in other script)
    Local::
        fieldi,fieldj (function): Holds the ith and jth wave mode function calculator
        r(float): The radius of the fibre (there is no need to calculate infinities as the definition might give you)
        int1,int2,int3,top bottom (float vectors [4,4]): Integrals (look at Agrawal for the integrals themselves)
    Returns::
        The first overlap integrals
    """
    if i == j:

        if  B[i] == 'LP01':
            return 1/161
        elif B[i] == 'LP11':
            return 1/170
    r = 62.45
    fieldi = B_func[i]
    fieldj = B_func[j]
    int1 = lambda y,x : np.abs(fieldi(y,x,w_vec))**2 * np.abs(fieldj(y,x,w_vec))**2
    top = dblquad(int1,-r,r,lambda x : -r,lambda x: r)[0]

    int2 = lambda y,x : np.abs(fieldi(y,x,w_vec))**2
    int3 = lambda y,x : np.abs(fieldj(y,x,w_vec))**2
    bottom = dblquad(int2,-r,r,lambda x : -r,lambda x: r)[0]*\
            dblquad(int3,-r,r,lambda x : -r,lambda x: r)[0]
    return top/bottom


def over2(i,j,k,l,B,B_func,w_vec):
    """
    Calculates the second overlap integral. If it is found that it is with itself then the
    inverse effective area is returned otherwise the integrals are calculated. For the mode calculations
    the hermit-gaussian approximation is taken.
    Also the calculation is done in terms of microns^2 and is transformed in to m^2 in calc_overlaps
    Inputs::
        i,j (int,int): Integer on what whave the overlap is calculated for
        B(str vec shape[4]): Holding the mode for each wave. (lp01 or lp11)
        B_func( function vec shape[4]) : Points to what function is used to calculate each mode(field0 or field1)
        w_vec(float vec shape[2]) : The width of the lp01 or the lp11 modes. (calculated in other script)
    Local::
        fieldi,fieldj (function): Holds the ith and jth wave mode function calculator
        r(float): The radius of the fibre (there is no need to calculate infinities as the definition might give you)
        int1,int2,int3,top bottom (float vectors [4,4]): Integrals (look at Agrawal for the integrals themselves)
    Returns::
        The second overlap integrals
    """
    if len(set([i,j,k,l])) == 1:
        if B[i] == 'LP01':
            return 1/161
        elif B[i] == 'LP11':
            return 1/170
    r = 62.45
    fieldi = B_func[i]
    fieldj = B_func[j]
    fieldk = B_func[k]
    fieldl = B_func[l]

    int1 = lambda y,x : fieldi(y,x,w_vec) * fieldj(y,x,w_vec) *fieldk(y,x,w_vec) * fieldl(y,x,w_vec)
    top = dblquad(int1,-r,r,lambda x : -r,lambda x: r)[0]
    bottom = 1
    for bot in B_func:
        int2 = lambda y,x : np.abs(bot(y,x,w_vec))**2
        bottom *= dblquad(int2,-r,r,lambda x : -r,lambda x: r)[0]
    bottom **=0.5
    return top/bottom


def inv_group_disp(i,lamda_c):

  if i == 0 or i == 2 :
      print(i,'LP01')
      #coeff = np.copy(mat_lp['LP01'])
      b01 = 0#np.copy(coeff[0][2])*1e-3
      DD = 19.8 *1e-6#np.copy(coeff[0][1])*1e6
      SS = 0.105*1e3#np.copy((coeff[0][0])*1e15)
  elif i == 1 or i == 3:
    print(i,'LP11')
    #coeff = np.copy(mat_lp['LP11'])
    b01 = -98e-15#np.copy(coeff[0][2])*1e-3
    DD = 21.8*1e-6#np.copy(coeff[0][1])*1e6
    SS = 0.108*1e3#np.copy((coeff[0][0])*1e15)
  #print(b01,DD,SS)
  return b01, -lamda_c**2*DD/(2*pi*c),lamda_c**4*SS/(4*(pi*c)**2) + lamda_c**3*DD/(2*(pi*c)**2)


def Dk_func(omega,lamda_c,B,zeroing,mat_lp):
    if zeroing == 4:
        return 0
    else:
        omega_c = 2*pi*c/lamda_c
        print(lamda_c)
        print(omega_c)
        b = np.zeros(4)
        D_vec = [19.8 *1e-6,21.8*1e-6,19.8 *1e-6,21.8*1e-6]
        S_vec = [0.105*1e3,0.108*1e3,0.105*1e3,0.108*1e3]
        beta2 = [-lamda_c**2*DD/(2*pi*c) for DD in D_vec]
        beta3 = []
        for iii,DD in enumerate(D_vec):
          SS = S_vec[iii]
          beta3.append(lamda_c**4*SS/(4*(pi*c)**2) + lamda_c**3*DD/(2*(pi*c)**2))
        print beta3
        for i in range(4):
            #dbeta1, beta2, beta3 = inv_group_disp(i,lamda_c)
            #b[i] = dbeta1 * (omega[i] - omega_c) + (1./2) * beta2 * (omega[i] - omega_c)**2 + \
            #        (1./6)*beta3*(omega[i]-omega_c)**3
            b[i] = (1./2) * beta2[i] * (omega[i] - omega_c)**2 + (1./6)*beta3[i]*(omega[i]-omega_c)**3
        Dk = b[2] + b[3] - b[0] - b[1]
        Dk += (-98e-3)*1e-12*(omega[0] - omega[2])
        #print(omega[2] - omega[0])
        return Dk

"""
def effective_phase_matching(lams,n2,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lami,dz,num_steps,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2):
    Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])
    for l, lamp1_ in enumerate(lamp1):
        for m,lamp2_ in enumerate(lamp2):
            for n,lams_ in enumerate(lams):

                lam_vec = np.array([lamp1_,lamp2_,lams_,1])
                omega = 2*pi*c/(lam_vec[:])
                omega[3] = omega[0] + omega[1] -omega[2]
                #lami.append(2*pi*c/omega[3])
                Dk_vec[l,m,n] = Dk_func(omega,lamda_c,B,zeroing,mat_lp)
                Dk = Dk_vec[l,m,n]
                A = np.array([(P_vec1)**0.5, (P_vec2)**0.5, P_signal_vec**0.5,0])
                #print n2*omega[3]*(xpm_spm(0,A,overlap1)+xpm_spm(1,A,overlap1)- xpm_spm(2,A,overlap1))/c
                Dk += n2*omega[3]*(xpm_spm(0,A,overlap1)+xpm_spm(1,A,overlap1)- xpm_spm(2,A,overlap1))/c
                Dk_vec[l,m,n] = Dk
    return Dk

def effective_phase_matching_general(lams,n2,P_vec1,P_vec2,P_signal_vec,lamp1,lamp2,lami,dz,num_steps,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2):
    Dk_vec = np.zeros([len(lamp1),len(lamp2),len(lams)])
    Dk_vec_nl = np.zeros([len(lamp1),len(lamp2),len(lams)])
    #AB_final =0

    for l, lamp1_ in enumerate(lamp1):
        for m,lamp2_ in enumerate(lamp2):
            for n,lams_ in enumerate(lams):
                print n,l,m
                lam_vec = np.array([lamp1_,lamp2_,lams_,1])
                omega = 2*pi*c/(lam_vec[:])
                omega[3] = omega[0] + omega[1] -omega[2]
                lami.append(2*pi*c/omega[3])
                Dk_vec[l,m,n] = Dk_func(omega,lamda_c,B,zeroing,mat_lp)
                #Dk = Dk_vec[l,m,n]
                A = np.array([(P_vec1)**0.5, (P_vec2)**0.5, P_signal_vec**0.5,0])
                #print n2*omega[3]*(xpm_spm(0,A,overlap1)+xpm_spm(1,A,overlap1)- xpm_spm(2,A,overlap1))/c
                Dk_vec_nl[l,m,n]= n2*omega[3]*(xpm_spm(0,A,overlap1)+xpm_spm(1,A,overlap1)- xpm_spm(2,A,overlap1))/c
                
    return Dk_vec,Dk_vec_nl,lami
"""

def FWM(n2,AB_final,P_vec1,P_vec2,P_signal_vec,lamp1,lams_,n,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor,seeds,num_eval,idl):


  P_signal = P_signal_vec[0]
  P1,P2 = P_vec1[0],P_vec2[0]
  if idl == 'pc':
    AB0 = np.array([P1, P2, P_signal,0], dtype='complex')
    lam_vec = np.array([lamp1,lamp2,lams_,1])

  else:
    AB0 = np.array([P_signal, P2, P1 ,0], dtype='complex')
    lam_vec = np.array([lams_,lamp2,lamp1,1])

  
  # 
  omega = 2*pi*c/lam_vec[:]
  omega[3] = omega[0] + omega[1] - omega[2]
  lami.append(2*pi*c/omega[3])
  

  Dk = Dk_func(omega,lamda_c,B,zeroing,mat_lp)*0.5

  ####Step size:
  dz = 1/(np.abs(Dk)*10)
  if Dk == 0 or dz > 1000:
     dz =0.1
  int_method = 'dopri5'
  AB0[:] = AB0[:]**0.5

  AB,outcome = integrator(int_method,AB0,zmin,zmax,dz,omega,Dk,n2,overlap1,overlap2)
  
  #if outcome==False:
  #    print('first try failed, trying adaptive steping...')
  #    exits = 0
  #    int_method = 'dop853'
  #    while exits<=55 and outcome == False:
  #        AB_final[:,i,j,k,l,m,n],outcome = \
  #                       integrator(int_method,AB0,zmin,zmax,dz,omega,Dk,n2,overlap1,overlap2)
  #        exits +=1
  #        dz *= 0.5
  #        print 'failed, trying step size:', dz,'...'
  #    if outcome == False:
  #          sys.exit('All integrations failed')
  return AB,lami


"""
def FWM_monte(n2,AB_final,Dk_vec,P_vec1,P_vec2,P_signal_vec,lamp1,lams_,n,lamp2,lami,dz,lamda_c,mat_lp,B,zmin,zmax,zeroing,overlap1,overlap2,noise_floor,seeds,num_eval,idl):
    for l, lamp1_ in enumerate(lamp1):
        for m,lamp2_ in enumerate(lamp2):
            lam_vec = np.array([lamp1_,lamp2_,lams_,1])
            if idl == 'bs':
                lam_vec[0],lam_vec[2] = lam_vec[2], lam_vec[0]
                #P_vec1,P_signal_vec = P_signal_vec,P_vec1
            omega = 2*pi*c/(lam_vec[:])
            omega[3] = omega[0] + omega[1] - omega[2]
            lami.append(2*pi*c/omega[3])

            Dk = Dk_func(omega,lamda_c,B,zeroing,mat_lp)
            dz = 1/(np.abs(Dk)*10)
            if Dk == 0 or dz > 1000:
                dz =0.1

            for i,P1 in enumerate(P_vec1):
                for j,P2 in enumerate(P_vec2):
                    #for k, P_signal in enumerate(P_signal_vec):
                        P_signal = P_signal_vec[n]

                        np.random.seed(seeds+n+l+i+j+k+m)
                        azimuth = np.random.rand(num_eval)*2*np.pi
                        AB_monte = np.zeros([4,num_eval],dtype=np.complex)
                        for t,theta in enumerate(azimuth):
                            AB0 = np.array([P1,P2, P_signal, 0], dtype='complex')
                            AB0[:] +=noise_floor
                            AB0[3] *= np.exp(1j*theta)
                            AB0[:] = AB0[:]**0.5
                            int_method = 'dopri5'
                            AB_monte[:,t],outcome = \
                                            integrator(int_method,AB0,zmin,zmax,dz,omega,Dk,n2,overlap1,overlap2)
                            if outcome==False:
                                print('first try failed, trying adaptive steping...')
                                exits = 0
                                int_method = 'dop853'
                                while exits<=55 and outcome == False:
                                    AB_monte[:,t],outcome = \
                                                   integrator(int_method,AB0,zmin,zmax,dz,omega,Dk,n2,overlap1,overlap2)
                                    exits +=1
                                    dz *= 0.5
                                    print 'failed, trying step size:', dz,'...'
                                if outcome == False:
                                    sys.exit('All integrations failed')

                        
                        for tt in range(4):
                            AB_final[tt,i,j,k,l,m,n] = np.average(AB_monte[tt,:])
        return AB_final,lami,Dk_vec
"""
