from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

import simulator2 as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

_Cl = 1e-15
_Cr = 1e-14
_Cc = 1e-12
para = sim.SimulationParameters(
    Cl=_Cl, Cr=_Cr,
    R1=1e6, L1=1e-9, C1=_Cc - _Cl - _Cr,
    R2=1e6, L2=1e-9, C2=_Cc - _Cr,
    R0=50.,
)
para.para.stiff_equation = True

# para.set_josephson()

# df = para.f01_d / para.Q1_d  # Hz
df = 5e6  # Hz
AMP = 1.  # V
PHASE = 0.  # rad
width = 100e6  # Hz
f_array = np.linspace(para.f01_d - width / 2, para.f01_d + width / 2, 101)

actual_f_array = np.zeros_like(f_array)
resp0_array = np.zeros_like(f_array, dtype=np.complex128)
resp1_array = np.zeros_like(f_array, dtype=np.complex128)
resp2_array = np.zeros_like(f_array, dtype=np.complex128)

# Run first time to get initial condition
fd_, df_ = para.tune(f_array[0], df, priority='f', regular=True)
para.set_df(df_)
para.set_drive_lockin([fd_], [AMP], [PHASE])
para.set_Nbeats(5)
sol = para.simulate()

para.set_Nbeats(2)
t_start = time.time()
for ii, fd in enumerate(f_array):
    print(ii)
    fd_, df_ = para.tune(fd, df, priority='f', regular=True)
    nd_ = int(round(fd_ / df_))
    para.set_df(df_)
    para.set_drive_lockin([fd_], [AMP], [PHASE])
    # para.set_noise_T(300.)
    sol = para.simulate(continue_run=True)

    V0 = sol[-para.ns:, 0]
    V0_fft = np.fft.rfft(V0) / para.ns
    V1 = sol[-para.ns:, 2]
    V1_fft = np.fft.rfft(V1) / para.ns
    V2 = sol[-para.ns:, 4]
    V2_fft = np.fft.rfft(V2) / para.ns
    actual_f_array[ii] = fd_
    resp0_array[ii] = V0_fft[nd_]
    resp1_array[ii] = V1_fft[nd_]
    resp2_array[ii] = V2_fft[nd_]
t_end = time.time()
t_tot = t_end - t_start
print("Total run took {:s}.".format(sim.format_sec(t_tot)))

G0 = para.tf0(f_array)
G1 = para.tf1(f_array)
G2 = para.tf2(f_array)

# fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
# ax1, ax2 = ax
# ax1.axvline(1e-9 * para.f01_b, ls='--', c='tab:blue')
# ax1.axvline(1e-9 * para.f02_b, ls='--', c='tab:orange')
# ax1.axvline(1e-9 * para.f01_d, ls=':', c='tab:blue')
# ax1.axvline(1e-9 * para.f02_d, ls=':', c='tab:orange')
# ax1.semilogy(1e-9 * f_array, np.abs(G1), '-', c='tab:blue')
# ax1.semilogy(1e-9 * actual_f_array, 2. * np.abs(resp1_array) / AMP, '.', c='tab:blue', label='cavity V1')
# ax1.semilogy(1e-9 * f_array, np.abs(G2), '-', c='tab:orange')
# ax1.semilogy(1e-9 * actual_f_array, 2. * np.abs(resp2_array) / AMP, '.', c='tab:orange', label='qubit V2')

# ax2.axvline(1e-9 * para.f01_b, ls='--', c='tab:blue', label='bare cavity')
# ax2.axvline(1e-9 * para.f02_b, ls='--', c='tab:orange', label='bare qubit')
# ax2.axvline(1e-9 * para.f01_d, ls=':', c='tab:blue', label='dressed cavity')
# ax2.axvline(1e-9 * para.f02_d, ls=':', c='tab:orange', label='dressed qubit')
# ax2.plot(1e-9 * f_array, np.angle(G1), '-', c='tab:blue', label='linear cavity')
# ax2.plot(1e-9 * actual_f_array, np.angle(resp1_array) - PHASE, '.', c='tab:blue')
# ax2.plot(1e-9 * f_array, np.angle(G2), '-', c='tab:orange', label='linear qubit')
# ax2.plot(1e-9 * actual_f_array, np.angle(resp2_array) - PHASE, '.', c='tab:orange')

# ax1.set_ylabel(r"Amplitude")
# ax2.set_ylabel(r"Phase [$\mathrm{rad}$]")
# ax2.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
# ax1.legend()
# ax2.legend(ncol=2)
# fig.show()

DOUBLE_PLOT = False
if DOUBLE_PLOT:
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True)
    ax0, ax1, ax2 = ax
    ax0r, ax1r, ax2r = ax0.twinx(), ax1.twinx(), ax2.twinx()
    axr = ax0r, ax1r, ax2r
    for ax_ in ax:
        ax_.spines['left'].set_color('tab:blue')
        ax_.tick_params(axis='y', which='both', colors='tab:blue')
        ax_.yaxis.label.set_color('tab:blue')
        # ticks = [1e9 * nn / para.df for nn in range(para.Nbeats + 1)]
        # ticks_minor = [1e9 * (nn + 0.5) / para.df for nn in range(para.Nbeats)]
        # ax_.set_xticks(ticks)
        # ax_.set_xticks(ticks_minor, minor=True)
        # for TT in ticks:
        #     ax_.axvline(TT, ls='--', c='tab:gray')
    for ax_ in axr:
        ax_.set_ylim(-np.pi, np.pi)
        ax_.spines['right'].set_color('tab:orange')
        ax_.tick_params(axis='y', which='both', colors='tab:orange')
        ax_.yaxis.label.set_color('tab:orange')
        ax_.set_yticks([-np.pi, 0., np.pi])
        ax_.set_yticks([-np.pi / 2, np.pi / 2], minor=True)
        ax_.set_yticklabels([u'\u2212\u03c0', '0', u'\u002b\u03c0'])
else:
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True)
    figr, axr = plt.subplots(3, 1, sharex=True, tight_layout=True)
    ax0, ax1, ax2 = ax
    ax0r, ax1r, ax2r = axr

# ax1.axvline(1e-9 * para.f01, ls='--', c='tab:blue')
# ax1.axvline(1e-9 * para.f02, ls='--', c='tab:orange')
ax0.semilogy(1e-9 * f_array, np.abs(G0), '-', c='tab:blue')
ax0.semilogy(1e-9 * actual_f_array, 2. * np.abs(resp0_array) / AMP, '.', c='tab:blue', label=r'$V_0$')
ax1.semilogy(1e-9 * f_array, np.abs(G1), '-', c='tab:blue')
ax1.semilogy(1e-9 * actual_f_array, 2. * np.abs(resp1_array) / AMP, '.', c='tab:blue', label=r'$V_1$')
ax2.semilogy(1e-9 * f_array, np.abs(G2), '-', c='tab:blue')
ax2.semilogy(1e-9 * actual_f_array, 2. * np.abs(resp2_array) / AMP, '.', c='tab:blue', label=r'$V_2$')

# ax2.axvline(1e-9 * para.f01, ls='--', c='tab:blue', label='bare cavity')
# ax2.axvline(1e-9 * para.f02, ls='--', c='tab:orange', label='bare qubit')
ax0r.plot(1e-9 * f_array, np.angle(G0), '-', c='tab:orange')
ax0r.plot(1e-9 * actual_f_array, np.angle(resp0_array) - PHASE, '.', c='tab:orange')
ax1r.plot(1e-9 * f_array, np.angle(G1), '-', c='tab:orange')
ax1r.plot(1e-9 * actual_f_array, np.angle(resp1_array) - PHASE, '.', c='tab:orange')
ax2r.plot(1e-9 * f_array, np.angle(G2), '-', c='tab:orange')
ax2r.plot(1e-9 * actual_f_array, np.angle(resp2_array) - PHASE, '.', c='tab:orange')

for ax_ in ax:
    ax_.set_ylabel(r"Amplitude")
    ax_.legend()
for ax_ in axr:
    ax_.set_ylabel(r"Phase")
ax2.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
# ax2.legend(ncol=2)
fig.show()
if not DOUBLE_PLOT:
    ax2r.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
    figr.show()
