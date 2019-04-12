import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, Planck
from scipy.signal import periodogram

import simulator as sim

Navg = 100

para = sim.SimulationParameters(
    Cl=0.333e-12, Cr=0.666e-12,
    R1=100., L1=1e-9, C1=1e-9,
    R0=50., R2=100.,
    fs=3e9,
)
w0, Q = para.calculate_resonance()
f0 = w0 / (2. * np.pi)

# Tnoise1 = Planck * f0 / (2. * Boltzmann)
# Tnoise0 = 0.
# Tnoise2 = 0.

Tnoise1 = 0.
Tnoise0 = Planck * f0 / (2. * Boltzmann)
Tnoise2 = 0.

# Tnoise1 = 0.
# Tnoise0 = 0.
# Tnoise2 = Planck * f0 / (2. * Boltzmann)

# Tnoise1 = Planck * f0 / (2. * Boltzmann)
# Tnoise0 = Planck * f0 / (2. * Boltzmann)
# Tnoise2 = Planck * f0 / (2. * Boltzmann)


df_ = 150e3  # Hz
_, df = para.tune(0., df_, regular=True)
para.set_df(df)
para.set_drive_none()


# ## TRADITIONAL WAY ###
print("Run tradtional way")

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
theoP1 = sigma0 * np.abs(para.tfn0P1(freqs))**2 + sigma1 * np.abs(para.tfn1P1(freqs))**2 + sigma2 * np.abs(para.tfn2P1(freqs))**2

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


# ## FASTER WAY ###
print("Run faster way")

psdV0_fast = np.zeros_like(freqs)
psdP1_fast = np.zeros_like(freqs)
psdV1_fast = np.zeros_like(freqs)
psdV2_fast = np.zeros_like(freqs)

t0 = time.time()
for ii in range(Navg):
    PSDv0_twosided = 2. * Boltzmann * para.noise_T0 * para.R0
    PSDv1_twosided = 2. * Boltzmann * para.noise_T1 * para.R1
    PSDv2_twosided = 2. * Boltzmann * para.noise_T2 * para.R2

    Vn0 = np.sqrt(PSDv0_twosided) * np.sqrt(para.fs) * np.random.randn(para.ns)
    Vn1 = np.sqrt(PSDv1_twosided) * np.sqrt(para.fs) * np.random.randn(para.ns)
    Vn2 = np.sqrt(PSDv2_twosided) * np.sqrt(para.fs) * np.random.randn(para.ns)

    Vn0_fft = np.fft.rfft(Vn0) / para.ns
    Vn1_fft = np.fft.rfft(Vn1) / para.ns
    Vn2_fft = np.fft.rfft(Vn2) / para.ns

    V0_fft = para.tfn00(freqs) * Vn0_fft + para.tfn10(freqs) * Vn1_fft + para.tfn20(freqs) * Vn2_fft
    V1_fft = para.tfn01(freqs) * Vn0_fft + para.tfn11(freqs) * Vn1_fft + para.tfn21(freqs) * Vn2_fft
    V2_fft = para.tfn02(freqs) * Vn0_fft + para.tfn12(freqs) * Vn1_fft + para.tfn22(freqs) * Vn2_fft
    P1_fft = para.tfn0P1(freqs) * Vn0_fft + para.tfn1P1(freqs) * Vn1_fft + para.tfn2P1(freqs) * Vn2_fft

    V0 = np.fft.irfft(V0_fft) * para.ns
    V1 = np.fft.irfft(V1_fft) * para.ns
    V2 = np.fft.irfft(V2_fft) * para.ns
    P1 = np.fft.irfft(P1_fft) * para.ns

    mf, X = periodogram(V0, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV0_fast += X
    mf, X = periodogram(P1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdP1_fast += X
    mf, X = periodogram(V1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV1_fast += X
    mf, X = periodogram(V2, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    psdV2_fast += X
t1 = time.time()
print("Computation time: {}".format(sim.format_sec(t1 - t0)))

psdV0_fast /= Navg
psdP1_fast /= Navg
psdV1_fast /= Navg
psdV2_fast /= Navg

ax.loglog(freqs, psdV0_fast, label='V0_fast', c='tab:pink')
ax.loglog(freqs, psdV1_fast, label='V1_fast', c='tab:gray')
ax.loglog(freqs, psdV2_fast, label='V2_fast', c='tab:olive')

ax.legend()
fig.canvas.draw()


fig2, ax2 = plt.subplots(tight_layout=True)
ax2.loglog(freqs, psdP1, label='P1', c='tab:orange')
ax2.loglog(freqs, psdP1_fast, label='P1_fast', c='tab:gray')
ax2.loglog(freqs, theoP1, '--', c='tab:cyan')
ax2.legend()
fig2.show()
