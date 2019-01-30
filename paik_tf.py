from __future__ import division, print_function

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

import simulator as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

# f_array = np.linspace(6e9, 9e9, 100000)
# f_array = np.linspace(7.958e9, 7.960e9, 100000)
# f_array = np.linspace(7.955e9, 7.957e9, 100000)
f_array = np.linspace(6.59e9, 6.60e9, 100000)
para = sim.SimulationParameters(
    Cl=1e-16, Cr=6.19e-15,
    R1=2.50e8, L1=9.94e-10, C1=3.98e-13,
    # R2=6.24e8, L2=7.75e-9, C2=6.44e-14,  # ground, 01
    R2=6.24e8, L2=8.22e-9, C2=6.44e-14,  # excited, 12
)

G1 = para.tf1(f_array)
G2 = para.tf2(f_array)

fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax1, ax2 = ax
ax1.axvline(1e-9 * para.f01_d, ls='--', c='tab:blue')
ax1.axvline(1e-9 * para.f02_d, ls='--', c='tab:orange')
ax1.semilogy(1e-9 * f_array, np.abs(G1), '-', c='tab:blue')
ax1.semilogy(1e-9 * f_array, np.abs(G2), '-', c='tab:orange')

ax2.axvline(1e-9 * para.f01_d, ls='--', c='tab:blue', label='bare cavity')
ax2.axvline(1e-9 * para.f02_d, ls='--', c='tab:orange', label='bare qubit')
ax2.plot(1e-9 * f_array, np.angle(G1), '-', c='tab:blue', label='linear cavity')
ax2.plot(1e-9 * f_array, np.angle(G2), '-', c='tab:orange', label='linear qubit')

ax1.set_ylabel(r"Amplitude")
ax2.set_ylabel(r"Phase [$\mathrm{rad}$]")
ax2.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
ax2.legend(ncol=2)
fig.show()
