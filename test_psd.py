from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann
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

# Tnoise1 = 30e-3
# Tnoise2 = 0.
# Tnoise0 = 0.

# Tnoise1 = 0.
# Tnoise2 = 30e-3
# Tnoise0 = 0.

# Tnoise1 = 0.
# Tnoise2 = 0.
# Tnoise0 = 30e-3

Tnoise1 = 30e-3
Tnoise2 = 30e-3
Tnoise0 = 30e-3

para = sim.SimulationParameters(
    Cl=1e-11, Cr=1e-10,
    R1=100., L1=1e-9, C1=1e-9,
    R2=100., L2=2e-9, C2=2e-9,
    fs=30e9,
)

df_ = 1e6  # Hz
_, df = para.tune(0., df_)
para.set_df(df)
para.set_drive_none()

# Run once to get initial condition
para.set_Nbeats(5)
para.set_noise_T(Tnoise1, Tnoise2, Tnoise0)
sol = para.simulate()

para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psd0 = np.zeros_like(freqs)
psd1 = np.zeros_like(freqs)
psd2 = np.zeros_like(freqs)

t0 = time.time()
for ii in range(Navg):
    print(ii)
    para.set_noise_T(Tnoise1, Tnoise2, Tnoise0)  # regenerate noise
    sol = para.simulate(continue_run=True)
    V0 = sol[-para.ns:, 0]
    V1 = sol[-para.ns:, 2]
    V2 = sol[-para.ns:, 4]
    mf, X = periodogram(V0, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psd0 += X
    mf, X = periodogram(V1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psd1 += X
    mf, X = periodogram(V2, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psd2 += X
t1 = time.time()
print("Simulation time: {}".format(sim.format_sec(t1 - t0)))

psd0 /= Navg
psd1 /= Navg
psd2 /= Navg

sigma1 = 4. * Boltzmann * para.noise_T1 * para.R1
sigma2 = 4. * Boltzmann * para.noise_T2 * para.R2
sigma0 = 4. * Boltzmann * para.noise_T0 * para.R0
theo1 = sigma1 * np.abs(para.tfn11(freqs))**2 + sigma2 * np.abs(para.tfn21(freqs))**2 + sigma0 * np.abs(para.tfn01(freqs))**2
theo2 = sigma1 * np.abs(para.tfn12(freqs))**2 + sigma2 * np.abs(para.tfn22(freqs))**2 + sigma0 * np.abs(para.tfn02(freqs))**2
theo0 = sigma1 * np.abs(para.tfn10(freqs))**2 + sigma2 * np.abs(para.tfn20(freqs))**2 + sigma0 * np.abs(para.tfn00(freqs))**2

fig, ax = plt.subplots(tight_layout=True)
ax.loglog(freqs, theo1, '--', c='tab:blue')
ax.loglog(freqs, theo2, '--', c='tab:orange')
ax.loglog(freqs, theo0, '--', c='tab:green')
ax.loglog(freqs, psd1, label='V1', c='tab:blue')
ax.loglog(freqs, psd2, label='V2', c='tab:orange')
ax.loglog(freqs, psd0, label='V0', c='tab:green')
ax.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax.legend()
fig.show()
