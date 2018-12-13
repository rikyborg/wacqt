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
Tnoise = 30e-3

para = sim.SimulationParameters(
    Cl=1e-15, Cr=1e-12,
    R1=3162., L1=1e-9, C1=1e-12,
    R2=3162., L2=2e-9, C2=2e-12,
)

df_ = 1e6  # Hz
_, df = para.tune(0., df_)
para.set_df(df)
para.set_drive_none()

# Run once to get initial condition
para.set_Nbeats(5)
para.set_noise_T(Tnoise)
sol = para.simulate()

para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psd0 = np.zeros_like(freqs)
psd1 = np.zeros_like(freqs)
psd2 = np.zeros_like(freqs)

t0 = time.time()
for ii in range(Navg):
    print(ii)
    para.set_noise_T(Tnoise, 0.)  # regenerate noise
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

fig, ax = plt.subplots(tight_layout=True)
ax.axhline(4. * Boltzmann * para.noise_T1 * para.R1, ls='--', c='tab:blue')
ax.axhline(4. * Boltzmann * para.noise_T2 * para.R2, ls='--', c='tab:orange')
ax.axhline(4. * Boltzmann * para.noise_T0 * para.R0, ls='--', c='tab:green')
ax.loglog(freqs, psd1, label='cavity V1', c='tab:blue')
ax.loglog(freqs, psd2, label='qubit V2', c='tab:orange')
ax.loglog(freqs, psd0, label='V0', c='tab:green')
ax.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax.legend()
fig.show()
