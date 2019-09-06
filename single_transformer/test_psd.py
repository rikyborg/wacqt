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

res, para = sim.SimulationParameters.from_measurement_single(
    2. * np.pi * 1e9, 1e4, 1e2,
    R0=50., R2=25., fs=20e9,
)
w0, Q = para.calculate_resonance()
f0 = w0 / (2. * np.pi)

# Tnoise0 = Planck * f0 / (2. * Boltzmann)
# Tnoise1 = 0.
# Tnoise2 = 0.

# Tnoise0 = 0.
# Tnoise1 = Planck * f0 / (2. * Boltzmann)
# Tnoise2 = 0.

# Tnoise0 = 0.
# Tnoise1 = 0.
# Tnoise2 = Planck * f0 / (2. * Boltzmann)

Tnoise0 = Planck * f0 / (2. * Boltzmann)
Tnoise1 = Planck * f0 / (2. * Boltzmann)
Tnoise2 = Planck * f0 / (2. * Boltzmann)


df_ = f0 / Q / 10.  # Hz
_, df = para.tune(0., df_, regular=True)
para.set_df(df)
para.set_drive_none()

# Run once to get initial condition
para.set_Nbeats(1)
para.set_noise_T(T1=Tnoise1, T0=Tnoise0, T2=Tnoise2)
sol = para.simulate()

para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psdI0 = np.zeros_like(freqs)
psdP1 = np.zeros_like(freqs)
psdV1 = np.zeros_like(freqs)
psdV2 = np.zeros_like(freqs)

t0 = time.time()
for ii in range(Navg):
    print(ii)
    para.set_noise_T(T1=Tnoise1, T0=Tnoise0, T2=Tnoise2)  # regenerate noise
    sol = para.simulate(continue_run=True)
    I0 = sol[-para.ns:, 0]
    P1 = sol[-para.ns:, 1]
    V1 = sol[-para.ns:, 2]
    V2 = para.calculate_V2(I0)
    mf, X = periodogram(I0, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdI0 += X
    mf, X = periodogram(P1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdP1 += X
    mf, X = periodogram(V1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV1 += X
    mf, X = periodogram(V2, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV2 += X
t1 = time.time()
print("Simulation time: {}".format(sim.format_sec(t1 - t0)))

psdI0 /= Navg
psdP1 /= Navg
psdV1 /= Navg
psdV2 /= Navg

sigma0 = 4. * Boltzmann * para.noise_T0 * para.R0
sigma1 = 4. * Boltzmann * para.noise_T1 * para.R1
sigma2 = 4. * Boltzmann * para.noise_T2 * para.R2
theoI0 = sigma0 * np.abs(para.tfn0I0(freqs))**2 + sigma1 * np.abs(para.tfn1I0(freqs))**2 + sigma2 * np.abs(para.tfn2I0(freqs))**2
theoP1 = sigma0 * np.abs(para.tfn0P1(freqs))**2 + sigma1 * np.abs(para.tfn1P1(freqs))**2 + sigma2 * np.abs(para.tfn2P1(freqs))**2
theoV1 = sigma0 * np.abs(para.tfn01(freqs))**2 + sigma1 * np.abs(para.tfn11(freqs))**2 + sigma2 * np.abs(para.tfn21(freqs))**2
theoV2 = sigma0 * np.abs(para.tfn02(freqs))**2 + sigma1 * np.abs(para.tfn12(freqs))**2 + sigma2 * np.abs(para.tfn22(freqs))**2

fig, ax = plt.subplots(tight_layout=True)
ax.loglog(freqs, psdI0, label='I0')
ax.loglog(freqs, psdP1, label='P1')
ax.loglog(freqs, psdV1, label='V1')
ax.loglog(freqs, psdV2, label='V2')
ax.loglog(freqs, theoI0, '--', label='I0')
ax.loglog(freqs, theoP1, '--', label='P1')
ax.loglog(freqs, theoV1, '--', label='V1')
ax.loglog(freqs, theoV2, '--', label='V2')
ax.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax.legend()
fig.show()
