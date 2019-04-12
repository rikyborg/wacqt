from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, Boltzmann

import simulator as sim

_wc = 2. * np.pi * 6e9
_chi = 2. * np.pi * 2e6
_Qb = 1e6
# _kappa = 2. * np.pi * 37.5e6 / 1e2
_kappa = _chi / 10
_Ql = _wc / _kappa

res, para_g, para_e = sim.SimulationParameters.from_measurement_crit_in(_wc, _chi, _Qb, _Ql)

w_g, Q_g = para_g.calculate_resonance()
w_e, Q_e = para_e.calculate_resonance()
chi = 0.5 * (w_e - w_g)
w_c = 0.5 * (w_g + w_e)
kappa = 0.5 * (w_g / Q_g + w_e / Q_e)

Eph = hbar * w_c
Tph = 0.5 * Eph / Boltzmann
T0 = Tph  # K
T2 = T0

_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(chi) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)

# reach equilibrium
Nrelax = int(round(2. * np.pi * df / kappa))
para_g.set_Nbeats(3 * Nrelax)
para_e.set_Nbeats(3 * Nrelax)
para_g.set_noise_T(T1=Tph, T0=T0, T2=T2)
para_e.set_noise_T(T1=Tph, T0=T0, T2=T2)
para_g.set_drive_none()
para_e.set_drive_none()
para_g.simulate(print_time=True)
para_e.simulate(print_time=True)

# readout run
para_g.set_Nbeats(4)
para_e.set_Nbeats(4)
para_g.set_noise_T(T1=Tph, T0=T0, T2=T2)
para_e.set_noise_T(T1=Tph, T0=T0, T2=T2)

t = para_g.get_time_arr()
t_drive = para_g.get_drive_time_arr()

AMP = 10e-6  # V
carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
window = np.zeros_like(t_drive)
idx = np.s_[0 * para_g.ns:2 * para_g.ns]
window[idx] = np.sin(2. * np.pi * 0.5 * df * t_drive[idx])**2
drive = window * carrier

para_g.set_drive_V(drive)
para_e.set_drive_V(drive)

Vg = drive[:-1]

sol_g = para_g.simulate(continue_run=True, print_time=True)
V0_g = sol_g[:, 0]
P1_g = sol_g[:, 1]
V1_g = sol_g[:, 2]
V2_g = sol_g[:, 3]
Vr_g = V0_g - (Vg / 2.)
sol_e = para_e.simulate(continue_run=True, print_time=True)
V0_e = sol_e[:, 0]
P1_e = sol_e[:, 1]
V1_e = sol_e[:, 2]
V2_e = sol_e[:, 3]
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
V2_g_spectrum = np.fft.rfft(V2_g) / len(V2_g)
Vr_e_spectrum = np.fft.rfft(Vr_e) / len(Vr_e)
P1_e_spectrum = np.fft.rfft(P1_e) / len(P1_e)
V1_e_spectrum = np.fft.rfft(V1_e) / len(V1_e)
V2_e_spectrum = np.fft.rfft(V2_e) / len(V2_e)

Vg_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
Vr_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
P1_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V1_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V2_g_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
Vr_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
P1_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V1_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V2_e_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)

Vg_envelope_spectrum[karray - para_g.Nbeats * nc] = Vg_spectrum[karray]
Vr_g_envelope_spectrum[karray - para_g.Nbeats * nc] = Vr_g_spectrum[karray]
P1_g_envelope_spectrum[karray - para_g.Nbeats * nc] = P1_g_spectrum[karray]
V1_g_envelope_spectrum[karray - para_g.Nbeats * nc] = V1_g_spectrum[karray]
V2_g_envelope_spectrum[karray - para_g.Nbeats * nc] = V2_g_spectrum[karray]
Vr_e_envelope_spectrum[karray - para_e.Nbeats * nc] = Vr_e_spectrum[karray]
P1_e_envelope_spectrum[karray - para_e.Nbeats * nc] = P1_e_spectrum[karray]
V1_e_envelope_spectrum[karray - para_e.Nbeats * nc] = V1_e_spectrum[karray]
V2_e_envelope_spectrum[karray - para_e.Nbeats * nc] = V2_e_spectrum[karray]

Vg_envelope = np.fft.ifft(Vg_envelope_spectrum) * Npoints
Vr_g_envelope = np.fft.ifft(Vr_g_envelope_spectrum) * Npoints
P1_g_envelope = np.fft.ifft(P1_g_envelope_spectrum) * Npoints
V1_g_envelope = np.fft.ifft(V1_g_envelope_spectrum) * Npoints
V2_g_envelope = np.fft.ifft(V2_g_envelope_spectrum) * Npoints
Vr_e_envelope = np.fft.ifft(Vr_e_envelope_spectrum) * Npoints
P1_e_envelope = np.fft.ifft(P1_e_envelope_spectrum) * Npoints
V1_e_envelope = np.fft.ifft(V1_e_envelope_spectrum) * Npoints
V2_e_envelope = np.fft.ifft(V2_e_envelope_spectrum) * Npoints

nr_ph_g = 0.5 * 2. * np.abs(P1_g_envelope)**2 / para_g.L1 / (hbar * w_g)
nr_ph_e = 0.5 * 2. * np.abs(P1_e_envelope)**2 / para_e.L1 / (hbar * w_e)


fig1, ax1 = plt.subplots(4, 1, sharex=True, tight_layout=True)
ax11, ax12, ax13, ax14 = ax1

ax11.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vg_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vr_g_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vr_e_envelope))
ax13.plot(1e6 * t_envelope, nr_ph_g)
ax13.plot(1e6 * t_envelope, nr_ph_e)
ax14.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(V2_g_envelope))
ax14.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(V2_e_envelope))

ax11.set_title("Amplitude")
ax14.set_xlabel(r"Time [$\mathrm{\mu s}$]")
fig1.show()


fig2, ax2 = plt.subplots(4, 1, sharex=True, tight_layout=True)
ax21, ax22, ax23, ax24 = ax2

ax21.plot(1e6 * t_envelope, np.angle(Vg_envelope))
ax22.plot(1e6 * t_envelope, np.angle(Vr_g_envelope))
ax22.plot(1e6 * t_envelope, np.angle(Vr_e_envelope))
ax23.plot(1e6 * t_envelope, np.angle(V1_g_envelope))
ax23.plot(1e6 * t_envelope, np.angle(V1_e_envelope))
ax24.plot(1e6 * t_envelope, np.angle(V2_g_envelope))
ax24.plot(1e6 * t_envelope, np.angle(V2_e_envelope))

ax21.set_title("Phase")
ax24.set_xlabel(r"Time [$\mathrm{\mu s}$]")
fig2.show()
