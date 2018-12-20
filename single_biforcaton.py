from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

import simulator_single as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

para = sim.SimulationParameters(
    Cl=14e-15,
    R1=200000., L1=6e-9, C1=240e-15,
    fs=80e9,
)
df = 3e6  # Hz

amps = 1e-6 * np.linspace(0.5, 2, 32)
M = len(amps)
# AMP = 1e-6  # V
PHASE = 0.  # rad
fstart = para.f01_d * (1. - 10. / para.Q1_d)
fstop = para.f01_d * (1. + 10. / para.Q1_d)

N = 256
f_array1 = np.linspace(fstart, fstop, N)
f_array2 = np.linspace(fstop, fstart, N)
f_array = np.concatenate((f_array1, f_array2))

PHI0 = 2.067833831e-15  # Wb, magnetic flux quantum
fake_PHI0 = PHI0 * 20.  # junctions in series
# para.set_duffing((2. * np.pi / fake_PHI0)**2 / 6)
para.set_josephson(PHI0=fake_PHI0)

actual_f_array = np.zeros_like(f_array)
actual_df_array = np.zeros_like(f_array)
for ii, fd in enumerate(f_array):
    fd_, df_ = para.tune(fd, df, priority='f')
    actual_f_array[ii] = fd_
    actual_df_array[ii] = df_
resp1_array = np.zeros((M, 2 * N), dtype=np.complex128)

for jj, AMP in enumerate(amps):
    print(jj)
    # Run first time to get initial condition
    fd_, df_ = para.tune(f_array[0], df, priority='f')
    para.set_df(df_)
    para.set_drive_lockin([fd_], [AMP], [PHASE])
    para.set_Nbeats(5)
    sol = para.simulate()

    para.set_Nbeats(2)
    t_start = time.time()
    for ii in range(2 * N):
        # print(ii)
        fd_ = actual_f_array[ii]
        df_ = actual_df_array[ii]
        nd_ = int(round(fd_ / df_))
        para.set_df(df_)
        para.set_drive_lockin([fd_], [AMP], [PHASE])
        # para.set_noise_T(300.)
        sol = para.simulate(continue_run=True)
        V1 = sol[-para.ns:, 2]
        V1_fft = np.fft.rfft(V1) / para.ns
        resp1_array[jj, ii] = V1_fft[nd_]
    t_end = time.time()
    t_tot = t_end - t_start
    print("Total run took {:s}.".format(sim.format_sec(t_tot)))

    resp1_array[jj, :] *= 2. / (AMP * np.exp(1j * PHASE))

fig, ax = plt.subplots(tight_layout=True)
axr = ax.twinx()

ax.spines['left'].set_color('tab:blue')
ax.tick_params(axis='y', which='both', colors='tab:blue')
ax.yaxis.label.set_color('tab:blue')

axr.set_ylim(-np.pi, np.pi)
axr.spines['right'].set_color('tab:orange')
axr.tick_params(axis='y', which='both', colors='tab:orange')
axr.yaxis.label.set_color('tab:orange')
axr.set_yticks([-np.pi, 0., np.pi])
axr.set_yticks([-np.pi / 2, np.pi / 2], minor=True)
axr.set_yticklabels([u'\u2212\u03c0', '0', u'\u002b\u03c0'])

ax.plot(1e-9 * actual_f_array[:N], 20. * np.log10(np.abs(resp1_array[0, :N])), '.', c='tab:blue', label=r'$V_1$ up')
ax.plot(1e-9 * actual_f_array[N:], 20. * np.log10(np.abs(resp1_array[0, N:])), '.', c='tab:green', label=r'$V_1$ down')

axr.plot(1e-9 * actual_f_array[:N], np.angle(resp1_array[0, :N]), '.', c='tab:orange')
axr.plot(1e-9 * actual_f_array[N:], np.angle(resp1_array[0, N:]), '.', c='tab:red')

ax.set_ylabel(r"Amplitude [dB]")
ax.legend()
axr.set_ylabel(r"Phase [rad]")
ax.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
fig.show()


a_diff = np.abs(resp1_array[:, :N]) - np.abs(resp1_array[:, N:][:, ::-1])
amax = a_diff.max()
amin = a_diff.min()
highlim = max(abs(amin), abs(amax))
lowlim = -highlim

fig2, ax2 = plt.subplots(tight_layout=True)
im = ax2.imshow(
    a_diff,
    # cmap='RdBu',
    cmap='cividis',
    vmin=lowlim,
    vmax=highlim,
    origin='lower',
    aspect='auto',
    extent=(f_array[0] / 1e9, f_array[N] / 1e9, amps[0] * 1e6, amps[-1] * 1e6),
)
ax2.set_xlabel('Drive frequency [GHz]')
ax2.set_ylabel(r'Drive amplitude [$\mathrm{\mu V}$]')
cb = fig2.colorbar(im)
cb.set_label(r"$A_\uparrow - A_\downarrow$")
fig2.show()
