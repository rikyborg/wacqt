from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, Planck
from scipy.signal import periodogram

import simulator as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

Navg = 100

C = 1e-9
L = 1e-9
w0 = 1. / np.sqrt(C * L)
f0 = w0 / (2. * np.pi)

# Tnoise1 = Planck * f0 / (2. * Boltzmann)
# Tnoise0 = 0.
# Tnoise2 = 0.

# Tnoise1 = 0.
# Tnoise0 = Planck * f0 / (2. * Boltzmann)
# Tnoise2 = 0.

# Tnoise1 = 0.
# Tnoise0 = 0.
# Tnoise2 = Planck * f0 / (2. * Boltzmann)

Tnoise1 = Planck * f0 / (2. * Boltzmann)
Tnoise0 = Planck * f0 / (2. * Boltzmann)
Tnoise2 = Planck * f0 / (2. * Boltzmann)

para = sim.SimulationParameters(
    Cl=0.333e-12, Cr=0.666e-12,
    R1=100., L1=1e-9, C1=1e-9,
    R0=50., R2=100.,
    fs=3e9,
)

df_ = 150e3  # Hz
_, df = para.tune(0., df_, regular=True)
para.set_df(df)
para.set_drive_none()

# Run once to get initial condition
para.set_Nbeats(1)
para.set_noise_T(Tnoise1, Tnoise0, Tnoise2)
sol = para.simulate()

para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psdV0 = np.zeros_like(freqs)
psdP1 = np.zeros_like(freqs)
psdV1 = np.zeros_like(freqs)
psdV2 = np.zeros_like(freqs)

t0 = time.time()
for ii in range(Navg):
    print(ii)
    para.set_noise_T(Tnoise1, Tnoise0, Tnoise2)  # regenerate noise
    sol = para.simulate(continue_run=True)
    V0 = sol[-para.ns:, 0]
    P1 = sol[-para.ns:, 1]
    V1 = sol[-para.ns:, 2]
    V2 = sol[-para.ns:, 3]
    mf, X = periodogram(V0, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV0 += X
    mf, X = periodogram(P1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdP1 += X
    mf, X = periodogram(V1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV1 += X
    mf, X = periodogram(V2, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV2 += X
t1 = time.time()
print("Simulation time: {}".format(sim.format_sec(t1 - t0)))

psdV0 /= Navg
psdP1 /= Navg
psdV1 /= Navg
psdV2 /= Navg

sigma0 = 4. * Boltzmann * para.noise_T0 * para.R0
sigma1 = 4. * Boltzmann * para.noise_T1 * para.R1
sigma2 = 4. * Boltzmann * para.noise_T2 * para.R2
theo0 = sigma0 * np.abs(para.tfn00(freqs))**2 + sigma1 * np.abs(para.tfn10(freqs))**2 + sigma2 * np.abs(para.tfn20(freqs))**2
theo1 = sigma0 * np.abs(para.tfn01(freqs))**2 + sigma1 * np.abs(para.tfn11(freqs))**2 + sigma2 * np.abs(para.tfn21(freqs))**2
theo2 = sigma0 * np.abs(para.tfn02(freqs))**2 + sigma1 * np.abs(para.tfn12(freqs))**2 + sigma2 * np.abs(para.tfn22(freqs))**2

fig, ax = plt.subplots(tight_layout=True)
ax.loglog(freqs, psdV0, label='V0', c='tab:blue')
ax.loglog(freqs, psdV1, label='V1', c='tab:orange')
ax.loglog(freqs, psdV2, label='V2', c='tab:green')
ax.loglog(freqs, theo0, '--', c='tab:red')
ax.loglog(freqs, theo1, '--', c='tab:purple')
ax.loglog(freqs, theo2, '--', c='tab:brown')
ax.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax.legend()
fig.show()
