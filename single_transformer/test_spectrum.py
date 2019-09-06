from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

import simulator as sim

_wc = 2. * np.pi * 6e9
_kappa = 2. * np.pi * 200e3
_Qb = 1e7
_chi = _kappa / 1.
_Ql = _wc / _kappa
_AMP = 1e-6  # V, will be adjusted later to reach max_ph
max_ph = 10.
detuning = 10.
double = True

res, para_g, para_e = sim.SimulationParameters.from_measurement(
    _wc, _chi, _Qb, _Ql, fs=120e9)
assert res.success

w_g, Q_g = para_g.calculate_resonance()
w_e, Q_e = para_e.calculate_resonance()
chi = 0.5 * (w_e - w_g)
w_c = 0.5 * (w_g + w_e)
kappa = 0.5 * (w_g / Q_g + w_e / Q_e)

_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(detuning * kappa) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
fs = para_g.fs
ns = int(round(fs / df))
T = 1. / df
dt = 1. / fs

t = np.linspace(0., 100 * T, 100 * ns, endpoint=False)
freqs = np.fft.rfftfreq(len(t), dt)

carrier = np.cos(2. * np.pi * fc * t)
window_single = np.zeros_like(t)
window_double = np.zeros_like(t)
idx = np.s_[0 * ns:2 * ns]
w_m_single = 2 * np.pi * df / 4
w_m_double = 2 * np.pi * df / 2
window_single[idx] = np.sin(w_m_single * t[idx])
window_double[idx] = np.sin(w_m_double * t[idx])
drive_single = window_single * carrier
drive_double = window_double * carrier

triangle_single = np.zeros_like(window_single)
triangle_single[t < T] = t[t < T] / T
triangle_single[np.logical_and(t >= T, t < 2 * T)] = -t[np.logical_and(t >= T, t < 2 * T)] / T + 2
triangle_double = np.zeros_like(window_single)
triangle_double[t < T / 2] = t[t < T / 2] / (T / 2)
triangle_double[np.logical_and(t >= T / 2, t < T)] = -t[np.logical_and(t >= T / 2, t < T)] / (T / 2) + 2
triangle_double[np.logical_and(t >= T, t < 3 * T / 2)] = t[np.logical_and(t >= T, t < 3 * T / 2)] / (T / 2) - 2
triangle_double[np.logical_and(t >= 3 * T / 2, t < 2 * T)] = -t[np.logical_and(t >= 3 * T / 2, t < 2 * T)] / (T / 2) + 4

drive_single_t = drive_single * triangle_single
drive_double_t = drive_double * triangle_double

G1_g = np.abs(para_g.tf1(freqs))
G1_g /= np.max(G1_g)
G1_e = np.abs(para_e.tf1(freqs))
G1_e /= np.max(G1_e)
drive_single_fft = np.abs(np.fft.rfft(drive_single))
drive_single_fft /= np.max(drive_single_fft)
drive_double_fft = np.abs(np.fft.rfft(drive_double))
drive_double_fft /= np.max(drive_double_fft)
drive_single_t_fft = np.abs(np.fft.rfft(drive_single_t))
drive_single_t_fft /= np.max(drive_single_t_fft)
drive_double_t_fft = np.abs(np.fft.rfft(drive_double_t))
drive_double_t_fft /= np.max(drive_double_t_fft)

fig, ax = plt.subplots(2, 1, tight_layout=True)
ax1, ax2 = ax

# ax1.plot(t, drive_single)
# ax1.plot(t, drive_double)
# ax1.plot(t, triangle_single)
# ax1.plot(t, triangle_double)

ax2.semilogy(freqs, drive_single_fft)
ax2.semilogy(freqs, drive_double_fft)
ax2.semilogy(freqs, G1_g)
ax2.semilogy(freqs, G1_e)
ax2.semilogy(freqs, drive_single_t_fft)
ax2.semilogy(freqs, drive_double_t_fft)

fig.show()
