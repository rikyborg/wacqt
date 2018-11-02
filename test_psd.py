from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
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
para.set_noise_T(300.)
sol = para.simulate()

para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psd1 = np.zeros_like(freqs)
psd2 = np.zeros_like(freqs)

t0 = time.time()
for ii in range(Navg):
    print(ii)
    para.set_noise_T(300.)  # regenerate noise
    sol = para.simulate(continue_run=True)
    V1 = sol[-para.ns:, 1]
    V2 = sol[-para.ns:, 3]
    mf, X = periodogram(V1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psd1 += X
    mf, X = periodogram(V2, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psd2 += X
t1 = time.time()
print("Simulation time: {}".format(sim.format_sec(t1 - t0)))

psd1 /= Navg
psd2 /= Navg

fig, ax = plt.subplots(tight_layout=True)
ax.loglog(freqs, psd1, label='cavity V1')
ax.loglog(freqs, psd2, label='qubit V2')
ax.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax.legend()
fig.show()
