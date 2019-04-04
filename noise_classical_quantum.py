from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, Planck
from scipy.signal import periodogram

import simulator_single as sim

T = 30e-3
R = 50.

fs = 100e9
df = 100e3
ns = round(fs / df)
dt = 1. / fs
freqs = np.fft.rfftfreq(ns, dt)
psd = np.zeros_like(freqs)

Navg = 200

para = sim.SimulationParameters(
    1e-12, 1e6, 1e-9, 1e-9,
    50., fs,
)
para.set_Nbeats(1)
para.set_df(df)

for ii in range(Navg):
    print(ii)
    # noise = np.sqrt(2. * Boltzmann * T * R) * np.sqrt(fs) * np.random.randn(ns)
    # noise = np.sqrt(fs) * np.random.randn(ns)
    # noise_fft = np.fft.rfft(noise) * np.sqrt(0.5 * 2. * R * Planck * freqs / np.tanh(Planck * freqs / (2. * Boltzmann * T)))
    # noise_fft[0] = 0.
    # noise = np.fft.irfft(noise_fft)
    para.set_noise_T(T, quantum=True)
    noise = para.noise0_array[:-2]
    f, Pxx = periodogram(noise, fs, detrend=False, return_onesided=True)
    psd += Pxx
psd /= Navg
psd /= 2  # two-sided

# freqs = np.logspace(0, 11, 1101)

classical = 2. * Boltzmann * T * R * np.ones_like(freqs)
quantum = 0.5 * 2. * R * Planck * freqs  # average positive and negative
both = 0.5 * 2. * R * Planck * freqs / np.tanh(Planck * freqs / (2. * Boltzmann * T))  # two-sided

fig, ax = plt.subplots(tight_layout=True)
ax.loglog(freqs, psd, '.', c='tab:blue')
ax.loglog(freqs, classical, '--', c='tab:orange')
ax.loglog(freqs, quantum, '--', c='tab:green')
ax.loglog(freqs, both, '--', c='tab:red')
fig.show()
