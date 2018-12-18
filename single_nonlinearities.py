from __future__ import division, print_function

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann, Planck

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
para.set_Nbeats(9)

AMP = 1e-6
fc, df = para.tune(para.f01_d, para.f01_d / para.Q1_d, priority='f')
para.set_df(df)
para.set_drive_lockin(
    [fc - df, fc, fc + df],
    [0.25 * AMP, 0.5 * AMP, 0.25 * AMP],
    [np.pi, 0., np.pi],
)
# para.set_drive_none()
# noise_T = Planck * para.f01_b / Boltzmann / 2
noise_T = 10e-9
# para.set_noise_T(noise_T, 0.)
# para.set_duffing(1e25)

sol_linear = para.simulate(print_time=True)

PHI0 = 2.067833831e-15  # Wb, magnetic flux quantum
fake_PHI0 = PHI0 * 20.
para.set_duffing((2. * np.pi / fake_PHI0)**2 / 6)
sol_duffing = para.simulate(print_time=True)

para.set_josephson(PHI0=fake_PHI0)
sol_josephson = para.simulate(print_time=True)

t = np.linspace(0., para.T, para.ns, endpoint=False)
freqs = np.fft.rfftfreq(4 * para.ns, para.dt)


fig1, ax1 = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax11, ax12, ax13 = ax1

ax11.plot(1e9 * t, sol_linear[-para.ns:, 0] * 1e6, label='linear')
ax11.plot(1e9 * t, sol_duffing[-para.ns:, 0] * 1e6)
ax11.plot(1e9 * t, sol_josephson[-para.ns:, 0] * 1e6)

ax12.plot(1e9 * t, sol_linear[-para.ns:, 1] / fake_PHI0)
ax12.plot(1e9 * t, sol_duffing[-para.ns:, 1] / fake_PHI0, label='Duffing')
ax12.plot(1e9 * t, sol_josephson[-para.ns:, 1] / fake_PHI0)

ax13.plot(1e9 * t, sol_linear[-para.ns:, 2] * 1e6)
ax13.plot(1e9 * t, sol_duffing[-para.ns:, 2] * 1e6)
ax13.plot(1e9 * t, sol_josephson[-para.ns:, 2] * 1e6, label='Josephson')

ax13.set_xlabel("Time [ns]")
ax11.set_ylabel(r"$V_0$ [$\mathrm{\mu V}$]")
ax12.set_ylabel(r"$\Phi_1 / \Phi_0$")
ax13.set_ylabel(r"$V_1$ [$\mathrm{\mu V}$]")

ax11.legend(loc='upper right')
ax12.legend(loc='upper right')
ax13.legend(loc='upper right')

fig1.show()


fig2, ax2 = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax21, ax22, ax23 = ax2

ax21.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_linear[-4 * para.ns:, 0]) / (4 * para.ns)) * 1e6, label='linear')
ax21.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_duffing[-4 * para.ns:, 0]) / (4 * para.ns)) * 1e6)
ax21.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_josephson[-4 * para.ns:, 0]) / (4 * para.ns)) * 1e6)

ax22.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_linear[-4 * para.ns:, 1]) / (4 * para.ns)) / fake_PHI0)
ax22.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_duffing[-4 * para.ns:, 1]) / (4 * para.ns)) / fake_PHI0, label='Duffing')
ax22.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_josephson[-4 * para.ns:, 1]) / (4 * para.ns)) / fake_PHI0)

ax23.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_linear[-4 * para.ns:, 2]) / (4 * para.ns)) * 1e6)
ax23.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_duffing[-4 * para.ns:, 2]) / (4 * para.ns)) * 1e6)
ax23.semilogy(1e-9 * freqs, np.abs(np.fft.rfft(sol_josephson[-4 * para.ns:, 2]) / (4 * para.ns)) * 1e6, label='Josephson')

for ax_ in ax2:
    ax_.set_xlim(1e-9 * (fc - 26 * df), 1e-9 * (fc + 26 * df))

ax23.set_xlabel("Frequency [GHz]")
ax21.set_ylabel(r"$V_0$ [$\mathrm{\mu V}$]")
ax22.set_ylabel(r"$\Phi_1 / \Phi_0$")
ax23.set_ylabel(r"$V_1$ [$\mathrm{\mu V}$]")

ax21.legend(loc='upper right')
ax22.legend(loc='upper right')
ax23.legend(loc='upper right')

fig2.show()
