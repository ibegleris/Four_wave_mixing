# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:35:27 2015

@author: Ioannis Begleris
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from scipy.constants import pi, c
from scipy.integrate import ode
from scipy.io import loadmat
from scipy.integrate import dblquad
import sys
from peakdetect import peakdetect
plt.ioff()


def dbm_to_w(dBm):
    """This function converts a power given in dBm to a power given in W."""
    return 1e-3 * 10**((dBm) / 10.)


def w_to_dbm(W):
    """This function converts a power given in W to a power given in dBm."""
    return 10. * np.log10(W) + 30


def xpm_spm(i, A, overlap1):
    "calculates the XPM-SPM term in the FWM equations(Agrawal)"
    sums = 0
    for k in range(4):
        if k != i:
            sums += overlap1[i, k] * np.abs(A[k])**2
    sums *= 2
    sums += overlap1[i, i] * np.abs(A[i])**2
    return sums * A[i]


def integrand(i, A, zz, omega, Dk, n2, overlap1, overlap2):
    i_vec = np.array([[1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]])
    ii = i_vec[i, :]
    temp = 2 * overlap2[i] * np.conj(A[ii[0]]) * A[ii[1]] * A[ii[2]]
    temp += xpm_spm(i, A, overlap1)
    gama = n2 * omega / c
    return 1j * gama * temp


def system(zz, AB, omega, Dk, n2, overlap1, overlap2):
    try:
        temp = AB[0]
    except:
        temp = AB
        AB = np.copy(zz)
        zz = np.copy(temp)
        del temp
        pass
    dABdz = np.zeros(4, dtype='complex')
    A0 = AB[0] + AB[1] * 1j
    A1 = AB[2] + AB[3] * 1j
    A2 = AB[4] + AB[5] * 1j
    A3 = AB[6] + AB[7] * 1j
    A = np.array([A0, A1, A2, A3])

    for i in range(4):
        dABdz[i] = integrand(i, A, zz, omega[i], Dk, n2, overlap1, overlap2)

    return dABdz.view(np.float64)


def integrator(
        int_method,
        AB0,
        zmin,
        zmax,
        dz,
        AB_final,
        i,
        j,
        omega,
        Dk,
        n2,
        overlap1,
        overlap2):
    r = ode(system).set_integrator(int_method)
    r.f_params = (omega, Dk, n2, overlap1, overlap2,)
    r.set_initial_value(AB0.view(np.float64), np.float64(zmin))
    count = 1
    num_steps = (zmax - zmin) / dz + 1
    while count < num_steps and r.successful():
        r.integrate(r.t + dz)
        count += 1
    if r.successful():
        AB_final[:, i, j] = r.y.view(np.complex128)
    return AB_final[:, i, j], r.successful()


def field0(y, x, w_vec):
    w = w_vec[0]
    return np.exp(-(x**2 + y**2) / w**2)


def field1(y, x, w_vec):
    w = w_vec[1]
    return (2 * 2**0.5 * x / w) * np.exp(-(x**2 + y**2) / w**2)


def over1(i, j, B, B_func, w_vec):
    if i == j:
        if B[i] == 'lp01':
            return 1 / 161
        else:
            return 1 / 170
    r = 62.45
    fieldi = B_func[i]
    fieldj = B_func[j]
    int1 = lambda y, x: np.abs(
        fieldi(y, x, w_vec))**2 * np.abs(fieldj(y, x, w_vec))**2
    top = dblquad(int1, -r, r, lambda x: -r, lambda x: r)[0]

    int2 = lambda y, x: np.abs(fieldi(y, x, w_vec))**2
    int3 = lambda y, x: np.abs(fieldj(y, x, w_vec))**2
    bottom = dblquad(int2, -r, r, lambda x : -r, lambda x: r)[0] *\
        dblquad(int3, -r, r, lambda x: -r, lambda x: r)[0]
    return top / bottom


def over2(i, j, k, l, B, B_func, w_vec):
    if len(set([i, j, k, l])) == 1:
        if B[i] == 'lp01':
            return 1 / 161
        else:
            return 1 / 170
    r = 62.45
    fieldi = B_func[i]
    fieldj = B_func[j]
    fieldk = B_func[k]
    fieldl = B_func[l]

    int1 = lambda y, x: fieldi(
        y, x, w_vec) * fieldj(y, x, w_vec) * fieldk(y, x, w_vec) * fieldl(y, x, w_vec)
    top = dblquad(int1, -r, r, lambda x: -r, lambda x: r)[0]
    bottom = 1
    for bot in B_func:
        int2 = lambda y, x: np.abs(bot(y, x, w_vec))**2
        bottom *= dblquad(int2, -r, r, lambda x: -r, lambda x: r)[0]
    bottom **= 0.5
    return top / bottom


def inv_group_disp(i, x, B, mat_lp, lamda_c):
    x = 2 * pi * c / x
    x = x - lamda_c
    if B[i] == 'lp01':
        coeff = mat_lp['Lp01']
        DD = coeff[0][1]
    else:
        coeff = mat_lp['Lp11']
        DD = coeff[0][1]
    return coeff[0][0] + coeff[0][1] * x + coeff[0][2] * \
        x**2, -lamda_c**2 * DD / (2 * pi * c)


def Dk_func(omega, lamda_c, B, zeroing, mat_lp):
    if zeroing == 4:
        return 0
    else:
        omega_c = c * 2 * pi / lamda_c
        b = np.zeros(4)
        for i in range(4):
            inv_group, dispers = inv_group_disp(
                i, omega[i], B, mat_lp, lamda_c)
            inv_group *= -2 * pi * c / omega[i]**2
            b[i] = inv_group * (omega[i] - omega_c) + \
                0.5 * dispers * (omega[i] - omega_c)**2
        Dk = b[2] + b[3] - b[0] - b[1]
        return Dk


def FWM(
        n2,
        P_vec,
        P_signal_vec,
        lamp,
        lams,
        lami,
        dz,
        num_steps,
        lamda_c,
        mat_lp,
        B,
        zmin,
        zmax,
        zeroing,
        overlap1,
        overlap2):
    AB_final = np.zeros([4, len(P_vec), len(lams)], dtype='complex')
    for i, Po in enumerate(P_vec):
        P = Po
        for j, lams_ in enumerate(lams):
            P_signal = P_signal_vec[j]

            AB0 = np.array([P, P, P_signal, 0], dtype='complex')
            AB0[:] = AB0[:]**0.5

            lami_ = lami[j]
            lamp_ = lamp[j]
            lam_vec = np.array([lamp_, lamp_, lams_, lami_])

            omega = 2 * pi * c / (lam_vec[:])

            omega[3] = 2 * omega[0] - omega[2]
            lami[j] = 2 * pi * c / omega[3]

            Dk = Dk_func(omega, lamda_c, B, zeroing, mat_lp)

            int_method = 'dop853'
            AB_final[
                :, i, j], outcome = integrator(
                int_method, AB0, zmin, zmax, dz, AB_final, i, j, omega, Dk, n2, overlap1, overlap2)
            if not outcome:
                print('first try failed, trying adaptive steping...')
                exits = 0
                int_method = 'dop853'
                while exits <= 55:
                    AB_final[
                        :, i, j], outcome = integrator(
                        int_method, AB0, zmin, zmax, dz, AB_final, i, j, omega, Dk, n2)
                    exits += 1
                    dz *= 0.5
                    print 'failed, trying step size:', dz, '...'
                if not outcome:
                    sys.exit('All integrations failed')
    return AB_final


def plotting(P_vec, AB_final, lams, lami, P_signal_vec):
    fig = plt.figure()
    for i, pp in enumerate(w_to_dbm(P_vec)):
        D_p_sim = np.abs(AB_final[0, i, :])**2
        plt.plot(lams, w_to_dbm(D_p_sim), 'x', label='pp')
    plt.xlabel(r'$\lambda (m)$')
    plt.title('Pump')
    plt.ylabel(r'$P(dbm)$')
    plt.legend(loc=(1.01, 0.805))
    plt.savefig('Pump.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    for i, pp in enumerate(w_to_dbm(P_vec)):
        D_s_sim = np.abs(AB_final[2, i, :])**2
        plt.plot(lams, w_to_dbm(D_s_sim), 'x', label='diff')
    plt.xlabel(r'$\lambda (m)$')
    plt.title('Signal')
    plt.ylabel(r'$P(dbm)$')
    plt.legend(loc=(1.01, 0.805))
    plt.savefig('Signal.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    for i, pp in enumerate(w_to_dbm(P_vec)):
        D_i_sim = np.abs(AB_final[3, i, :])**2
        plt.plot(lami, w_to_dbm(D_i_sim), 'x', label='sim')
    plt.xlabel(r'$\lambda (m)$')
    plt.title('Idler')
    plt.ylabel(r'$P(dbm)$')
    plt.legend(loc=(1.01, 0.805))
    plt.savefig('Idler.png', bbox_inches='tight')
    plt.close(fig)

    DeltaE = P_signal_vec + 2 * P_vec * np.ones(len(P_signal_vec)) - \
        D_i_sim - 2 * D_p_sim - D_s_sim
    print 'Infinity norm of the energy difference', np.linalg.norm(DeltaE, np.inf)
    return 0


def calc_overlaps(B, w_vec):
    B_func = [0, 0, 0, 0]
    for i, b in enumerate(B):
        if b == 'lp01':
            B_func[i] = field0
        else:
            B_func[i] = field1
    ##############################
    overlap1 = np.zeros([4, 4])

    for i in range(4):
        for j in range(4):
            overlap1[i, j] = over1(i, j, B, B_func, w_vec)
    i_vec = np.array([[1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]])

    overlap2 = np.zeros(4)
    zeroing = 0
    for i in range(4):
        ii = i_vec[i, :]
        overlap2[i] = over2(i, ii[0], ii[1], ii[2], B, B_func, w_vec)
        if overlap2[i] == 0:
            zeroing += 1
    overlap1 *= 1e12
    overlap2 *= 1e12
    return overlap1, overlap2, zeroing


def input_powers_wavelengths():
    """From the experimental data
    """
    mat = loadmat('../loading_data/LP11_FWM_data.mat')
    lams_vec_exp = mat['lam_vals']
    D = mat['D']
    del mat
    lamp = np.zeros(len(D[0, :]))
    lams = np.zeros(len(D[0, :]))
    lami = np.zeros(len(D[0, :]))

    D_p = np.zeros(len(D[0, :]))
    D_s = np.zeros(len(D[0, :]))
    D_i = np.zeros(len(D[0, :]))
    for i in range(len(D[0, :])):
        _max, _min = peakdetect(D[:, i], lams_vec_exp[:, i], 50)
        max_ = np.asanyarray(_max)
        max_ = max_[np.argsort(max_[:, 1])]
        lamp[i], D_p[i] = max_[-1::, 0][0], max_[-1::, 1][0]
        lami[i], D_i[i] = max_[-3::3, 0][0], max_[-3::3, 1][0]
        lams[i], D_s[i] = max_[-2::2, 0][0], max_[-2::2, 1][0]

    D_p = D_p[0:-3:]
    D_s = D_s[0:-3:]
    D_i = D_i[0:-3:]

    lamp = lamp[0:-3:]
    lams = lams[0:-3:]
    lami = lami[0:-3:]

    P_vec = np.arange(22.7, 23.7, 2)
    P_vec += 10
    P_vec = dbm_to_w(P_vec)

    P_signal_vec = dbm_to_w(D_s) - dbm_to_w(D_i)

    lamp = np.copy(lamp) * 1e-9
    lams = np.copy(lams) * 1e-9
    lami = np.copy(lami) * 1e-9
    return P_vec, P_signal_vec, lamp, lams, lami


def main():
    # The constants
    n2 = 1.1225783990826979e-20  # nonlinear coefficient
    # the central freequency that the betas are calculated around
    lamda_c = 1.5508e-6
    ###############

    num_steps = 10  # initial number of steps for the integrator
    d = 1e3  # propagation distance

    zmin = 0  # starting point of the fibre
    zmax = d  # end point of interest
    # linearly spaced propagation vector
    z = np.linspace(zmin, zmax, num_steps)
    dz = 1 * (z[1] - z[0])
    num_steps = (zmax - zmin) / dz + 1

    # What beam is in what mode
    B = []
    B.append('lp01')  # pump1
    B.append('lp11')  # pump2
    B.append('lp01')  # sigal
    B.append('lp11')  # Idler
    #############

    w_vec = np.loadtxt('../loading_data/widths.dat')
    w_vec *= 1e6
    overlap1, overlap2, zeroing = calc_overlaps(B, w_vec)
    print('calculated the overlaps going for the ode')
    P_vec, P_signal_vec, lamp, lams, lami = input_powers_wavelengths()
    mat_lp = loadmat('../loading_data/coeffs.mat')
    AB_final = FWM(
        n2,
        P_vec,
        P_signal_vec,
        lamp,
        lams,
        lami,
        dz,
        num_steps,
        lamda_c,
        mat_lp,
        B,
        zmin,
        zmax,
        zeroing,
        overlap1,
        overlap2)

    plotting(P_vec, AB_final, lams, lami, P_signal_vec)
    return 0


if __name__ == '__main__':
    main()
