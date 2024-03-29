from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, Planck

import simulator_single as sim

para_g = sim.SimulationParameters(
    Cl=float.fromhex('0x1.d900307dad616p-44'),
    R1=float.fromhex('0x1.5134882b7a3a9p+19'),
    L1=float.fromhex('0x1.97a112542a4e2p-36'),
    C1=float.fromhex('0x1.a9f95ac362cecp-35'),
    R0=50.,
    fs=100e9,
)
para_e = sim.SimulationParameters(
    Cl=float.fromhex('0x1.d900307dad616p-44'),
    R1=float.fromhex('0x1.5134882b7a3a9p+19'),
    L1=float.fromhex('0x1.94f7301817a9dp-36'),
    C1=float.fromhex('0x1.a9ef70a3f4f56p-35'),
    R0=50.,
    fs=100e9,
)
w_g = float.fromhex('0x1.bc5de40ce9d27p+34')
w_e = float.fromhex('0x1.bdd8967667a65p+34')
Ql_g = float.fromhex('0x1.78a47cffcd8d8p+11')
Ql_e = float.fromhex('0x1.779cb183fdc2bp+11')

w_c = 0.5 * (w_e + w_g)
chi = 0.5 * (w_e - w_g)
kappa_g = w_g / Ql_g
kappa_e = w_e / Ql_e
kappa = 0.5 * (kappa_e + kappa_g)  # average kappa
Ql = w_c / kappa  # OBS: different than average Ql!

Eph = Planck / (2. * np.pi) * w_c
Tph = 0.5 * Eph / Boltzmann
T0 = 10e-3  # K

_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(chi) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)
para_g.set_Nbeats(6)
para_e.set_Nbeats(6)

# To get the init for noise below
# Nrelax = int(np.ceil(2. * np.pi * df / kappa * 5))
# para_g.set_Nbeats(Nrelax)
# para_g.set_drive_none()
# para_g.set_noise_T(Tph, T0)
# para_g.simulate()
# para_e.set_Nbeats(Nrelax)
# para_e.set_drive_none()
# para_e.set_noise_T(Tph, T0)
# para_e.simulate()
# print("para_g.next_init = np.array([")
# print("    float.fromhex('{:s}'),".format(para_g.next_init[0].hex()))
# print("    float.fromhex('{:s}'),".format(para_g.next_init[1].hex()))
# print("    float.fromhex('{:s}'),".format(para_g.next_init[2].hex()))
# print("])")
# print("para_e.next_init = np.array([")
# print("    float.fromhex('{:s}'),".format(para_e.next_init[0].hex()))
# print("    float.fromhex('{:s}'),".format(para_e.next_init[1].hex()))
# print("    float.fromhex('{:s}'),".format(para_e.next_init[2].hex()))
# print("])")
# raise RuntimeError
para_g.next_init = np.array([
    float.fromhex('-0x1.9febc1c27f532p-20'),
    float.fromhex('0x1.05ee96271d49dp-60'),
    float.fromhex('0x1.2f4c7aa8a0307p-26'),
])
para_e.next_init = np.array([
    float.fromhex('-0x1.3043237bc23b8p-21'),
    float.fromhex('0x1.a9b707cea9f99p-62'),
    float.fromhex('-0x1.4686bf0fa54fdp-27'),
])

t = para_g.get_time_arr()
t_drive = para_g.get_drive_time_arr()

AMP = 3e-6  # V
carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
window = np.zeros_like(t_drive)
idx = np.s_[2 * para_g.ns:4 * para_g.ns]
window[idx] = np.sin(2. * np.pi * 0.5 * df * t_drive[idx])**2
drive = window * carrier

para_g.set_drive_V(drive)
para_e.set_drive_V(drive)
para_g.set_noise_T(Tph, T0)
para_e.set_noise_T(Tph, T0)

Vg = drive[:-1]

sol_g = para_g.simulate(continue_run=True)  # use init from above
V0_g = sol_g[:, 0]
P1_g = sol_g[:, 1]
V1_g = sol_g[:, 2]
Vr_g = V0_g - (Vg / 2.)
sol_e = para_e.simulate(continue_run=True)  # use init from above
V0_e = sol_e[:, 0]
P1_e = sol_e[:, 1]
V1_e = sol_e[:, 2]
Vr_e = V0_e - (Vg / 2.)


Nimps = 31
Npoints = 1000
nc = int(round(fc / df))
karray = np.arange((nc - Nimps // 2) * para_g.Nbeats, (nc + Nimps // 2) * para_g.Nbeats + 1)
t_envelope = np.linspace(0., para_g.Nbeats / para_g.df, Npoints, endpoint=False)

Vg_spectrum = np.fft.rfft(Vg) / len(Vg)
Vr_g_spectrum = np.fft.rfft(Vr_g) / len(Vr_g)
P1_g_spectrum = np.fft.rfft(P1_g) / len(P1_g)
V1_g_spectrum = np.fft.rfft(V1_g) / len(V1_g)
Vr_e_spectrum = np.fft.rfft(Vr_e) / len(Vr_e)
P1_e_spectrum = np.fft.rfft(P1_e) / len(P1_e)
V1_e_spectrum = np.fft.rfft(V1_e) / len(V1_e)

Vg_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
Vr_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
P1_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V1_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
Vr_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
P1_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V1_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)

Vg_envelope_spectrum[karray - para_g.Nbeats * nc] = Vg_spectrum[karray]
Vr_g_envelope_spectrum[karray - para_g.Nbeats * nc] = Vr_g_spectrum[karray]
P1_g_envelope_spectrum[karray - para_g.Nbeats * nc] = P1_g_spectrum[karray]
V1_g_envelope_spectrum[karray - para_g.Nbeats * nc] = V1_g_spectrum[karray]
Vr_e_envelope_spectrum[karray - para_e.Nbeats * nc] = Vr_e_spectrum[karray]
P1_e_envelope_spectrum[karray - para_e.Nbeats * nc] = P1_e_spectrum[karray]
V1_e_envelope_spectrum[karray - para_e.Nbeats * nc] = V1_e_spectrum[karray]

Vg_envelope = np.fft.ifft(Vg_envelope_spectrum) * Npoints
Vr_g_envelope = np.fft.ifft(Vr_g_envelope_spectrum) * Npoints
P1_g_envelope = np.fft.ifft(P1_g_envelope_spectrum) * Npoints
V1_g_envelope = np.fft.ifft(V1_g_envelope_spectrum) * Npoints
Vr_e_envelope = np.fft.ifft(Vr_e_envelope_spectrum) * Npoints
P1_e_envelope = np.fft.ifft(P1_e_envelope_spectrum) * Npoints
V1_e_envelope = np.fft.ifft(V1_e_envelope_spectrum) * Npoints


fig1, ax1 = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax11, ax12, ax13 = ax1

ax11.plot(t_envelope, 2 * np.abs(Vg_envelope))
ax12.plot(t_envelope, 2 * np.abs(V1_g_envelope))
ax12.plot(t_envelope, 2 * np.abs(V1_e_envelope))
ax13.plot(t_envelope, 2 * np.abs(Vr_g_envelope))
ax13.plot(t_envelope, 2 * np.abs(Vr_e_envelope))

fig1.show()


fig2, ax2 = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax21, ax22, ax23 = ax2

ax21.plot(t_envelope, np.angle(Vg_envelope))
ax22.plot(t_envelope, np.angle(V1_g_envelope))
ax22.plot(t_envelope, np.angle(V1_e_envelope))
ax23.plot(t_envelope, np.angle(Vr_g_envelope))
ax23.plot(t_envelope, np.angle(Vr_e_envelope))

fig2.show()

"""








correct = 0
for ii in range(100):
    para_g.next_init = np.array([
        float.fromhex('-0x1.9febc1c27f532p-20'),
        float.fromhex('0x1.05ee96271d49dp-60'),
        float.fromhex('0x1.2f4c7aa8a0307p-26'),
    ])
    para_e.next_init = np.array([
        float.fromhex('-0x1.3043237bc23b8p-21'),
        float.fromhex('0x1.a9b707cea9f99p-62'),
        float.fromhex('-0x1.4686bf0fa54fdp-27'),
    ])
    para_g.set_noise_T(Tph, T0)
    para_e.set_noise_T(Tph, T0)
    
    sol_g = para_g.simulate(continue_run=True)  # use init from above
    V0_g = sol_g[:, 0]
    Vr_g = V0_g - (Vg / 2.)
    sol_e = para_e.simulate(continue_run=True)  # use init from above
    V0_e = sol_e[:, 0]
    Vr_e = V0_e - (Vg / 2.)
    
    Vr_g_spectrum = np.fft.rfft(Vr_g) / len(Vr_g)
    Vr_e_spectrum = np.fft.rfft(Vr_e) / len(Vr_e)
    
    Vr_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
    Vr_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
    
    Vr_g_envelope_spectrum[karray - para_g.Nbeats * nc] = Vr_g_spectrum[karray]
    Vr_e_envelope_spectrum[karray - para_e.Nbeats * nc] = Vr_e_spectrum[karray]
    
    Vr_g_envelope = np.fft.ifft(Vr_g_envelope_spectrum) * Npoints
    Vr_e_envelope = np.fft.ifft(Vr_e_envelope_spectrum) * Npoints
    
    sg = Vr_g_envelope[250:750]
    se = Vr_e_envelope[250:750]
    
    if np.real(np.sum(np.conj(tg - te) * sg)) > 0.:
        correct += 1
    if np.real(np.sum(np.conj(tg - te) * se)) < 0.:
        correct += 1
"""
