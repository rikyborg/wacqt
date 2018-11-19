from __future__ import division, print_function

import time

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

df = 5e6  # Hz
AMP = 1.  # V
PHASE = 0.  # rad
f_array = np.linspace(1e9, 5.5e9, 1000)
para = sim.SimulationParameters(
    Cl=1e-16, Cr=6.19e-15,
    R1=2.50e8, L1=9.94e-10, C1=3.98e-13,
    R2=6.24e8, L2=7.75e-9, C2=6.44e-14,
)
# para.set_duffing(1e23)
# para.set_josephson()

actual_f_array = np.zeros_like(f_array)
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

    V1 = sol[-para.ns:, 1]
    V1_fft = np.fft.rfft(V1) / para.ns
    V2 = sol[-para.ns:, 3]
    V2_fft = np.fft.rfft(V2) / para.ns
    actual_f_array[ii] = fd_
    resp1_array[ii] = V1_fft[nd_]
    resp2_array[ii] = V2_fft[nd_]
t_end = time.time()
t_tot = t_end - t_start
print("Total run took {:s}.".format(sim.format_sec(t_tot)))

G1 = para.tf1(f_array)
G2 = para.tf2(f_array)

fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax1, ax2 = ax
ax1.axvline(1e-9 * para.f01, ls='--', c='tab:blue')
ax1.axvline(1e-9 * para.f02, ls='--', c='tab:orange')
ax1.semilogy(1e-9 * f_array, np.abs(G1), '-', c='tab:blue')
ax1.semilogy(1e-9 * actual_f_array, 2. * np.abs(resp1_array) / AMP, '.', c='tab:blue', label='cavity V1')
ax1.semilogy(1e-9 * f_array, np.abs(G2), '-', c='tab:orange')
ax1.semilogy(1e-9 * actual_f_array, 2. * np.abs(resp2_array) / AMP, '.', c='tab:orange', label='qubit V2')

ax2.axvline(1e-9 * para.f01, ls='--', c='tab:blue', label='bare cavity')
ax2.axvline(1e-9 * para.f02, ls='--', c='tab:orange', label='bare qubit')
ax2.plot(1e-9 * f_array, np.angle(G1), '-', c='tab:blue', label='linear cavity')
ax2.plot(1e-9 * actual_f_array, np.angle(resp1_array) - PHASE, '.', c='tab:blue')
ax2.plot(1e-9 * f_array, np.angle(G2), '-', c='tab:orange', label='linear qubit')
ax2.plot(1e-9 * actual_f_array, np.angle(resp2_array) - PHASE, '.', c='tab:orange')

ax1.set_ylabel(r"Amplitude")
ax2.set_ylabel(r"Phase [$\mathrm{rad}$]")
ax2.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
ax1.legend()
ax2.legend(ncol=2)
fig.show()
