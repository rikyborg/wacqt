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

para = sim.SimulationParameters(
    Cl=1e-16, Cr=6.19e-15,
    R1=2.50e8, L1=9.94e-10, C1=3.98e-13,
    # R2=6.24e8, L2=7.75e-9, C2=6.44e-14,  # ground, 01
    R2=6.24e8, L2=8.22e-9, C2=6.44e-14,  # excited, 12
)

fc_ground = 7959147851.478515  # Hz
fc_excited = 7956492274.9227495  # Hz
df_ = fc_ground - fc_excited
fc_ = 0.5 * (fc_ground + fc_excited)
f1_ = fc_ - df_ / 2.
f1, df = para.tune(f1_, df_, priority='f', regular=True)
f_arr = np.array([f1, f1 + df])
A_arr = np.array([0.5, 0.5])
P_arr = np.array([0., np.pi])

para.set_df(df)
para.set_Nbeats(4)

t = para.get_time_arr()
t_drive = para.get_drive_time_arr()
drive = np.zeros_like(t_drive)
for ii in range(len(f_arr)):
    drive += A_arr[ii] * np.cos(2. * np.pi * f_arr[ii] * t_drive + P_arr[ii])
drive[:para.ns] = 0.
drive[-2 * para.ns:] = 0.
para.set_drive_V(drive)

para.set_noise_T(.03)
sol = para.simulate()

fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax1, ax2, ax3 = ax
for ax_ in ax:
    for nn in range(para.Nbeats + 1):
        TT = nn / para.df
        ax_.axvline(1e9 * TT, ls='--', c='tab:gray')
ax1.plot(1e9 * t_drive, drive, c='tab:blue', label='drive [V]')
ax2.plot(1e9 * t, 1e3 * sol[:, 1], c='tab:orange', label='Cavity V1 [mV]')
ax3.plot(1e9 * t, 1e3 * sol[:, 3], c='tab:green', label='Qubit V2 [mV]')
for ax_ in ax:
    ax_.legend()
ax3.set_xlabel("Time [ns]")
fig.show()