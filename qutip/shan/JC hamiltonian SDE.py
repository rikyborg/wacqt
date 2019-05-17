# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:04:14 2018

@author: Shan
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
import SDE_ito_algorithms as sde

#stochastic simulations of Jaynes-Cummings hamiltonian in the dispersive regime

#-----------------------------setting up simulation

#size of truncated Hilbert space of resonator
N = 20
#annihilation operator
a = qt.destroy(N)
#number operator
n = a.dag() * a

#operators for bipartite system
#qubit ins first slot, cavity in second
sigmaz_bi = qt.tensor(qt.sigmaz(), qt.qeye(N))
number_bi = qt.tensor(qt.qeye(2), n)
coupling = qt.tensor(qt.sigmaz(), n)

#parameters in GHz
omega_cavity = 6 * 2 * np.pi
omega_qubit = 4.8 * 2 * np.pi
delta = np.abs(omega_cavity - omega_qubit)
chi = 1.33 * 10**-3 * 2 * np.pi
g = np.sqrt(chi * delta)

#initial qubit and resonator state (density matrix)
initial_cavity = qt.fock(N, 0)
excited = qt.basis(2, 0)
ground = qt.basis(2, 1)
initial_qubit = excited  #(ground+excited)/np.sqrt(2)

initial_state = qt.tensor(initial_qubit, initial_cavity)
initial_state_dm = 0.5 * qt.ket2dm(initial_state) + 0.5 * qt.ket2dm(
    qt.tensor(ground, initial_cavity))

#will be working in a frame rotating (interaction frame) at the following frequency (GHz):
rotation_omega = omega_cavity
omega_drive = omega_cavity
omega_drive_eff = omega_drive - rotation_omega

#since sigmaz commutes with all operators in the hamiltonian, I can work also in a rotating qubit frame
rotating_qubit = omega_qubit


#hbar=1
#also ignoring the vacuum energy contribution
def H_disp(t):
    return (omega_cavity - rotation_omega) * number_bi + 0.5 * (
        (omega_qubit - rotating_qubit) + chi) * sigmaz_bi + chi * coupling


#time translated everything so the pulse is centered around t=0 ns
def drive(t):
    drive_amp_max = 50 * 10**-3  #GHz
    env_omega = chi / 2
    #time in ns
    stop_time = np.pi / env_omega

    if (t > -stop_time) & (t <= stop_time):
        return drive_amp_max * np.sin(env_omega * t)
    else:
        return 0


def H_drive(t):
    return drive(t) * qt.tensor(qt.qeye(2), a.dag()) + np.conjugate(
        drive(t)) * qt.tensor(qt.qeye(2), a)


#resonator loss in GHz
kappa = omega_cavity / (3.8 * 10**6)

#qubit losses in GHz
gamma_1 = 2 * np.pi * 1 / 100. * 10**-3
gamma_2 = 2 * np.pi * 1 / 500.
gamma_phi = gamma_2 - gamma_1 / 2

#time in units of ns
#time axis defined so that the pulse center lies at t=0 ns
time = np.linspace(-1500, 1500, 15000)

#list of dissipation operator sin Lindblad form
#dissipation_list=[np.sqrt(kappa)*qt.tensor(qt.qeye(2), a), np.sqrt(gamma_1)*qt.tensor(qt.sigmam(), qt.qeye(N)), np.sqrt(gamma_phi/2)*sigmaz_bi]
dissipation_list = [np.sqrt(kappa) * qt.tensor(qt.qeye(2), a)]

#----------------------------------- performing the simulation

rho_t2, wiener_incr = sde.rk_sde(H_disp, H_drive, initial_state_dm,
                                 dissipation_list, time)

#-------------------------------------analyzing results

#what is the homodyne current like?
x = (a + a.dag()) * np.sqrt(kappa)
x_bi = qt.tensor(qt.qeye(2), x)
J_meas_current = qt.expect(x_bi, rho_t2) + wiener_incr / (time[1] - time[0])

plt.figure(1)
ax = plt.subplot(111)
ax.plot(time, J_meas_current)
plt.title('Measured current I-quadrature')
plt.xlabel('time /ns')
plt.ylabel('I-quadrature')

#the jump is not visible in the "pure" current
#using some averaging (same conventions as in gambetta et.al.)
#integration time in ns
integration_time = 10

#linearly interpolate homodyne current
#linear_interp_J_current = interp1d(time, J_meas_current, kind='linear')

##define substeps to perform numerical integration over
#int_step=np.arange(0, integration_time*10**-6*2*np.pi, time[1]-time[0])
#if len(int_step)<20:
#    print("Warning: no. of integration step for output current is:")
#    print(len(int_step))
#
#
#int_current_time_step=np.arange(time[0], time[-1], integration_time*10**-6*2*np.pi)
#
#int_current=[simps(linear_interp_J_current(int_step + ct), int_step) for ct in int_current_time_step[0:-1]]/(integration_time*np.sqrt(kappa))
#
#plt.figure(2)
#ax=plt.subplot(111)
#ax.plot(int_current_time_step[0:-1], int_current)
#plt.title('integrated current I-quadrature')
#plt.xlabel('time /milliseconds')
#plt.ylabel('integrated I-quadrature')
#

#computing the Bloch vector components of the qubit
x_comp = qt.expect(rho_t2, qt.tensor(qt.sigmax(), qt.qeye(N)))
y_comp = qt.expect(rho_t2, qt.tensor(qt.sigmay(), qt.qeye(N)))
z_comp = qt.expect(rho_t2, qt.tensor(qt.sigmaz(), qt.qeye(N)))

fig = plt.figure(3)
ax = plt.subplot(111)
ax.plot(time, np.real(x_comp), label='x')
ax.plot(time, np.real(y_comp), label='y')
ax.plot(time, np.real(z_comp), label='z')
plt.title('Bloch vector components for qubit')
plt.xlabel('time /ns')
plt.ylabel('average x, y, z')
ax.legend()

cavity_I_quadrature = qt.expect(x_bi, rho_t2)
plt.figure(4)
ax = plt.subplot(111)
ax.plot(time, cavity_I_quadrature)
plt.title('Inside cavity I-quadrature')
plt.xlabel('time /ns')
plt.ylabel('I-quadrature')

cavity_photon_number = qt.expect(number_bi, rho_t2)
plt.figure(5)
ax = plt.subplot(111)
ax.plot(time, cavity_photon_number)
plt.title('mean cavity photon number')
plt.xlabel('time /ns')
plt.ylabel('photon number')

drive_amp = [drive(t) for t in time]
plt.figure(6)
ax = plt.subplot(111)
ax.plot(time, drive_amp)
plt.title('drive amp')
plt.xlabel('time /ns')
plt.ylabel('amp / GHz')

plt.show()
