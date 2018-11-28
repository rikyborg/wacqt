from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import periodogram
from scipy.constants import Boltzmann, Planck

import simulator as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

Navg = 10000
# /. cR -> 1*^-4 /. cS -> 1*^-3 /. l -> 10. /. r -> 1*^5
_Cl = 1e-18  # F
_Cr = 1e-13  # F
_Cs = 1e-12  # F
_L = 1e-8  # H
_R = 1e5  # ohm
para = sim.SimulationParameters(
    Cl=_Cl, Cr=_Cr,
    R1=_R, L1=_L, C1=_Cs - _Cl - _Cr,
    R2=_R, L2=_L, C2=_Cs - _Cr,
)

# AMP = 0.
AMP = 0.5

# para.set_josephson(which='right')
# para.set_josephson(which='both')
# para.set_josephson(PHI0=2.067833831e-15 * np.sqrt(10.), which='right')
para.set_josephson(PHI0=2.067833831e-15 * np.sqrt(10.), which='both')

fH_ = np.sqrt(1. / (_L * (_Cs - _Cr))) / (2. * np.pi)
fL_ = np.sqrt(1. / (_L * (_Cs + _Cr))) / (2. * np.pi)
QH = _R * np.sqrt((_Cs - _Cr) / _L)
QL = _R * np.sqrt((_Cs + _Cr) / _L)
center = 0.5 * (fH_ + fL_)
split = fH_ - fL_
width = _Cs / (_R * (_Cs**2 - _Cr**2)) / (2. * np.pi)  # average
noise_T = Planck * center / Boltzmann / 2

fP_ = center
df_ = np.sqrt(split * width)
# df_ = width
fP, df = para.tune(fP_, df_, priority='f', regular=True)
fL = int(round(fL_ / df)) * df
fH = int(round(fH_ / df)) * df
kP, kL, kH = int(round(fP / df)), int(round(fL / df)), int(round(fH / df))

para.set_df(df)
if AMP:
    para.set_drive_lockin(fP, AMP, 0.)
else:
    para.set_drive_none()

# Run once to get initial condition
para.set_Nbeats(5)
para.set_noise_T(noise_T)
sol = para.simulate()

para.set_Nbeats(1)
freqs = np.fft.rfftfreq(para.ns, para.dt)
psd1 = np.zeros_like(freqs)
psd2 = np.zeros_like(freqs)
resp_low = np.zeros(Navg, dtype=np.complex128)
resp_high = np.zeros(Navg, dtype=np.complex128)
karray = np.arange(int(kP - 1.5 * (kH - kL)), int(kP + 1.5 * (kH - kL)))
resp_zoom = np.zeros((len(karray), Navg), dtype=np.complex128)

t0 = time.time()
for ii in range(Navg):
    print(ii)
    para.set_noise_T(noise_T)  # regenerate noise
    sol = para.simulate(continue_run=True)
    V1 = sol[-para.ns:, 1]
    V2 = sol[-para.ns:, 3]
    # mf, X = periodogram(V1, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    # psd1 += X
    # mf, X = periodogram(V2, para.fs, window='boxcar', nfft=None, detrend=False, return_onesided=True, scaling='density')
    # psd2 += X
    V1_fft = np.fft.rfft(V1) / len(V1)
    psd1 += 2. * (V1_fft.real**2 + V1_fft.imag**2) / df
    V2_fft = np.fft.rfft(V2) / len(V2)
    psd2 += 2. * (V2_fft.real**2 + V2_fft.imag**2) / df
    resp_low[ii] = V1_fft[kL]
    resp_high[ii] = V1_fft[kH]
    resp_zoom[:, ii] = V1_fft[karray]
t1 = time.time()
print("Simulation time: {}".format(sim.format_sec(t1 - t0)))

psd1 /= Navg
psd2 /= Navg

idx_zoom = np.logical_and(freqs > center - 1.5 * split, freqs < center + 1.5 * split)
freqs_zoom = freqs[idx_zoom]
psd1_zoom = psd1[idx_zoom]
psd2_zoom = psd2[idx_zoom]

fig1, ax1 = plt.subplots(tight_layout=True)
ax1.semilogy(freqs_zoom, psd1_zoom, label='cavity V1')
ax1.semilogy(freqs_zoom, psd2_zoom, label='qubit V2')
ax1.set_xlabel(r"Frequency [$\mathrm{HZ}$]")
ax1.set_ylabel(r"PSD [$\mathrm{V}^2/\mathrm{HZ}$]")
ax1.set_title(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
ax1.legend()
# ax1.axvline(fP_, ls='--')
# ax1.axvline(fL_, ls='--')
# ax1.axvline(fH_, ls='--')
ax1.axvline(fP, ls='--')
ax1.axvline(fL, ls='--')
ax1.axvline(fH, ls='--')
fig1.show()

# fig2, ax2 = plt.subplots(3, 2, tight_layout=True)
# (ax21, ax22), (ax23, ax24), (ax25, ax26) = ax2
# ax21.hist2d(resp_low.real, resp_high.real, bins=100)
# ax22.hist2d(resp_low.real, resp_high.imag, bins=100)
# ax23.hist2d(resp_low.imag, resp_high.real, bins=100)
# ax24.hist2d(resp_low.imag, resp_high.imag, bins=100)
# ax25.hist2d(resp_low.real, resp_low.imag, bins=100)
# ax26.hist2d(resp_high.real, resp_high.imag, bins=100)
# for ax_ in fig2.get_axes():
#     ax_.set_aspect('equal')
# ax21.set_title(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
# fig2.show()

I_low = (resp_low.real - resp_low.real.mean()) / resp_low.real.std()
Q_low = (resp_low.imag - resp_low.imag.mean()) / resp_low.imag.std()
I_high = (resp_high.real - resp_high.real.mean()) / resp_high.real.std()
Q_high = (resp_high.imag - resp_high.imag.mean()) / resp_high.imag.std()

fig3, ax3 = plt.subplots(3, 2, tight_layout=True)
(ax31, ax32), (ax33, ax34), (ax35, ax36) = ax3
ax31.hist2d(I_low, I_high, bins=100, range=[[-5, 5], [-5, 5]])
ax31.set_xlabel(r"$I_\mathrm{L}$")
ax31.set_ylabel(r"$I_\mathrm{H}$")
ax32.hist2d(I_low, Q_high, bins=100, range=[[-5, 5], [-5, 5]])
ax32.set_xlabel(r"$I_\mathrm{L}$")
ax32.set_ylabel(r"$Q_\mathrm{H}$")
ax33.hist2d(Q_low, I_high, bins=100, range=[[-5, 5], [-5, 5]])
ax33.set_xlabel(r"$Q_\mathrm{L}$")
ax33.set_ylabel(r"$I_\mathrm{H}$")
ax34.hist2d(Q_low, Q_high, bins=100, range=[[-5, 5], [-5, 5]])
ax34.set_xlabel(r"$Q_\mathrm{L}$")
ax34.set_ylabel(r"$Q_\mathrm{H}$")
ax35.hist2d(I_low, Q_low, bins=100, range=[[-5, 5], [-5, 5]])
ax35.set_xlabel(r"$I_\mathrm{L}$")
ax35.set_ylabel(r"$Q_\mathrm{L}$")
ax36.hist2d(I_high, Q_high, bins=100, range=[[-5, 5], [-5, 5]])
ax36.set_xlabel(r"$I_\mathrm{H}$")
ax36.set_ylabel(r"$Q_\mathrm{H}$")
for ax_ in fig3.get_axes():
    ax_.set_aspect('equal')
    ax_.set_xticks([])
    ax_.set_yticks([])
fig3.suptitle(r"$A_\mathrm{P}$ = " + "{:.1e} V".format(AMP))
fig3.show()


filename = "double_AMP_{:.1e}_V_df_{:.1e}_Hz_Navg_{:d}".format(AMP, df, Navg)
np.savez(
    filename,
    para=para.pickable_copy(),

    freqs=freqs,
    psd1=psd1,
    psd2=psd2,
    resp_low=resp_low,
    resp_high=resp_high,
    karray=karray,
    resp_zoom=resp_zoom,

    AMP=AMP,
    fH_=fH_,
    fL_=fL_,
    QH=QH,
    QL=QL,
    center=center,
    split=split,
    width=width,
    noise_T=noise_T,
    fP_=fP_,
    df_=df_,
    fP=fP,
    df=df,
    fL=fL,
    fH=fH,
)
