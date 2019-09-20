from __future__ import absolute_import, division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

from simulators import sim_transformer as sim
# from simulators import sim_notch as sim
# from simulators import sim_reflection as sim
# from simulators import sim_transmission as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

_, para = sim.SimulationParameters.from_measurement_single(2. * np.pi * 1e9, 1e4, 1e2, fs=20e9)
df = 1e6  # Hz

w0, Q = para.calculate_resonance()
f0 = w0 / (2. * np.pi)
bw = f0 / Q

AMP = 0.5e-6  # V
PHASE = 0.  # rad
fstart = f0 * (1. - 10. / Q)
fstop = f0 * (1. + 10. / Q)

f_array = np.linspace(fstart, fstop, 501)

PHI0 = 2.067833831e-15  # Wb, magnetic flux quantum
fake_PHI0 = PHI0 * 20.  # junctions in series
# para.set_duffing((2. * np.pi / fake_PHI0)**2 / 6)
# para.set_josephson(PHI0=fake_PHI0)

actual_f_array = np.zeros_like(f_array)
resp_array = np.zeros((sim.NEQ, len(f_array)), dtype=np.complex128)

# Run first time to get initial condition
fd_, df_ = para.tune(f_array[0], df, priority='f')
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

    for jj in range(sim.NEQ):
        x = sol[-para.ns:, jj]
        x_fft = np.fft.rfft(x) / para.ns
        resp_array[jj, ii] = x_fft[nd_]
    actual_f_array[ii] = fd_
t_end = time.time()
t_tot = t_end - t_start
print("Total run took {:s}.".format(sim.format_sec(t_tot)))

resp_array *= 2. / (AMP * np.exp(1j * PHASE))

f_plot = np.linspace(fstart, fstop, 20001)
transfer_function = np.zeros((sim.NEQ, len(f_plot)), np.complex128)
for ii in range(sim.NEQ):
    transfer_function[ii, :] = para.state_variable_tf(f_plot, ii)


fig, ax = plt.subplots(2, sim.NEQ, sharex=True, tight_layout=True)
ax1, ax2 = ax

for _ax in ax.flatten():
    _ax.axvline(1e-9 * f0, ls='--', c='tab:gray', alpha=0.5)
    _ax.axvline(1e-9 * (f0 - bw / 2), ls='--', c='tab:gray', alpha=0.25)
    _ax.axvline(1e-9 * (f0 + bw / 2), ls='--', c='tab:gray', alpha=0.25)

for ii, _ax in enumerate(ax1):
    _ax.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(resp_array[ii, :])), '.')
    _ax.plot(1e-9 * f_plot, 20. * np.log10(np.abs(transfer_function[ii, :])), '--')
    _ax.set_title(para.state_variables_latex[ii])

for ii, _ax in enumerate(ax2):
    _ax.plot(1e-9 * actual_f_array, np.angle(resp_array[ii, :]), '.')
    _ax.plot(1e-9 * f_plot, np.angle(transfer_function[ii, :]), '--')

fig.show()
