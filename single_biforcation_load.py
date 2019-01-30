from __future__ import absolute_import, division, print_function

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

with np.load("single_biforcation_mp_result.npz") as npz:
    para = np.asscalar(npz['para'])
    amps = npz['amps']
    f_array = npz['f_array']
    df = np.asscalar(npz['df'])
    resp1_array = npz['resp1_array']

N = amps.shape[0]
actual_f_array = np.zeros_like(f_array)
actual_df_array = np.zeros_like(f_array)
for ii, fd in enumerate(f_array):
    fd_, df_ = para.tune(fd, df, priority='f')
    actual_f_array[ii] = fd_
    actual_df_array[ii] = df_


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

ax.plot(1e-9 * actual_f_array[:N], 20. * np.log10(np.abs(resp1_array[38, :N])), '.', c='tab:blue', label=r'$V_1$ up')
ax.plot(1e-9 * actual_f_array[N:], 20. * np.log10(np.abs(resp1_array[38, N:])), '.', c='tab:green', label=r'$V_1$ down')

axr.plot(1e-9 * actual_f_array[:N], np.angle(resp1_array[38, :N]), '.', c='tab:orange')
axr.plot(1e-9 * actual_f_array[N:], np.angle(resp1_array[38, N:]), '.', c='tab:red')

ax.set_ylabel(r"Amplitude [dB]")
ax.legend()
axr.set_ylabel(r"Phase [rad]")
ax.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
fig.show()


a_diff = np.abs(resp1_array[:, :N]) - np.abs(resp1_array[:, N:][:, ::-1])
# a_diff = 20. * np.log10(np.abs(resp1_array[:, :N])) - 20. * np.log10(np.abs(resp1_array[:, N:][:, ::-1]))
amax = a_diff.max()
amin = a_diff.min()
highlim = max(abs(amin), abs(amax))
lowlim = -highlim
# highlim = amax
# lowlim = amin

fig2, ax2 = plt.subplots(tight_layout=True)
im = ax2.imshow(
    a_diff,
    # cmap='RdBu',
    cmap='cividis',
    # cmap='viridis',
    # cmap='Greys',
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
ax2.axvline(para.f01_d / 1e9, ls='--', c='k')
ax2.plot(4.07152, 0.702565, 'x', c='tab:red', ms=9)
fig2.show()
