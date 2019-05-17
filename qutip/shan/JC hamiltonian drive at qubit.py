# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:15:44 2018

@author: Shan
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as inter


#JC hamiltonian, driving at the qubit frequency but coupled to the cavity
#what actually happens?

#size of truncated Hilbert space of resonator
N=60
#annihilation operator
a=qt.destroy(N)

#parameters in GHz
omega_cavity=6*2*np.pi
omega_qubit=4.8*2*np.pi
g=20*10**-3*2*np.pi
omega_drive=omega_qubit+g**2/(omega_qubit-omega_cavity)

drive_amp=0.7*2*np.pi

#resonator loss GHz
kappa=10*10**-3*2*np.pi

#qubit losses GHz
gamma_1=1/7.*10**-3*2*np.pi
gamma_2=1/500.*2*np.pi
gamma_phi=gamma_2-gamma_1/2



#time in units of ns
time=np.linspace(0,100,600)

#rotates at the qubit resonance frequency


initial_cavity=qt.fock(N, 0)
excited=qt.basis(2,0)
ground=qt.basis(2,1)
initial_qubit=ground

initial_state=qt.tensor(initial_cavity, initial_qubit)


#working in aframe rotating at the drive frequency for the cavity
#for the qubit, I am working at the qubit resonance
rotating_qubit=omega_qubit

#The JC hamiltonian with drive terms coupling drive to cavity modes, should I include drive coupled to qubit?

H_disp = (omega_qubit - rotating_qubit)/2*qt.tensor(qt.qeye(N), qt.sigmaz()) + (omega_cavity-omega_drive) * qt.tensor(a.dag()*a, qt.qeye(2)) - g * (qt.tensor(a, qt.sigmap()) + qt.tensor(a.dag(), qt.sigmam())) + drive_amp*qt.tensor(a.dag(), qt.qeye(2)) + np.conjugate(drive_amp)*qt.tensor(a, qt.qeye(2))

#I want to understand the qubit dynamics


#simulating dynamics with qubit losses
diss_list=[np.sqrt(kappa)*qt.tensor(a, qt.qeye(2)), np.sqrt(gamma_1)*qt.tensor(qt.qeye(N), qt.sigmam()), np.sqrt(gamma_phi/2)*qt.tensor(qt.qeye(N), qt.sigmaz())]
simul_diss=qt.mesolve(H_disp, initial_state, time, diss_list, [])
#tracing out the resonator, I want to keep the qubit
simulated_qubit=[simul_diss.states[index].ptrace(1) for index in np.arange(0,len(time))]
#tracing out the qubit, I want to keep the resonator
simulated_resonator=[simul_diss.states[index].ptrace(0) for index in np.arange(0,len(time))]

#computing the Bloch vector components of the qubit
x_comp=qt.expect(simulated_qubit, qt.sigmax())
y_comp=qt.expect(simulated_qubit, qt.sigmay())
z_comp=qt.expect(simulated_qubit, qt.sigmaz())

fig=plt.figure(1)
ax=plt.subplot(111)
ax.plot(time, x_comp, label='x')
ax.plot(time, y_comp, label='y')
ax.plot(time, z_comp, label='z')
plt.title('Bloch vector components for qubit')
plt.xlabel('time /nanoseconds')
plt.ylabel('average x, y, z')
ax.legend()


#cavity photon number
cav_no=qt.expect(simulated_resonator, a.dag()*a)

fig=plt.figure(2)
ax=plt.subplot(111)
ax.plot(time, cav_no)
plt.title('average cavity photon number')
plt.xlabel('time /nanoseconds')
plt.ylabel('average photons number')

plt.show()

