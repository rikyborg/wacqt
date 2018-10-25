from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

import simulator as sim

Navg = 1000

para = sim.SimulationParameters(
    Cl=1e-15, Cr=1e-12,
    R1=3162., L1=1e-9, C1=1e-12,
    R2=3162., L2=2e-9, C2=2e-12,
    w_arr=[0.], A_arr=[0.], P_arr=[0.],
)
para.set_Nbeats(5)
# para.set_duffing(1e23)
para.set_noise_T(300.)

df = 1e6  # Hz
_, df_ = para.tune(0., df, prio_f=False)
para.set_df(df_)

init = np.array([0., 0., 0., 0.])
sol = para.simulate(init=init)
para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psd1 = np.zeros_like(freqs)
psd2 = np.zeros_like(freqs)

for ii in range(Navg):
    print(ii)
    para.set_noise_T(300.)
    sol = para.simulate(init=init)
    V1 = sol[-para.ns:, 1]
    V2 = sol[-para.ns:, 3]
    # V1_fft = np.fft.rfft(V1) / len(V1)
    # psd += np.abs(V1_fft)**2
    mf, X = periodogram(V1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psd1 += X
    mf, X = periodogram(V2, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psd2 += X

psd1 /= Navg
psd2 /= Navg

fig, ax = plt.subplots(tight_layout=True)
ax.semilogy(freqs, psd1)
ax.semilogy(freqs, psd2)
ax.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
fig.show()
