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
    Cl=1e-15, Cr=1e-12,
    R1=3162., L1=1e-9, C1=1e-12,
    R2=3162., L2=2e-9, C2=2e-12,
    fs=1000e9,
)
# para.set_duffing(1e29)
# para.set_josephson()
# para.set_josephson(which='both')

df_ = 50e6  # Hz
fc_ = 4.04e9  # Hz
f1_ = fc_ - df_ / 2.
f1, df = para.tune(f1_, df_, priority='f', regular=True)
f_arr = np.array([f1, f1 + df])
A_arr = np.array([0.00005, 0.00005])
P_arr = np.array([0., np.pi])

Nr = 5  # nr windows to reach steady state (throw away)
Na = 4  # nr windows to FFT
para.set_Nbeats(Nr + Na)

para.set_df(df)
para.set_drive_lockin(f_arr, A_arr, P_arr)

# para.set_noise_T(3e-6)

para.para.stiff_equation = False
sol = para.simulate(print_time=True)

para.para.stiff_equation = True
sol_stiff = para.simulate(print_time=True)


t = para.get_time_arr()
t = t[-para.ns:]
t -= t[0]
freqs = np.fft.rfftfreq(Na * para.ns, para.dt)

V1 = sol[-Na * para.ns:, 1]
V1_fft = np.fft.rfft(V1) / len(V1)
V1_stiff = sol_stiff[-Na * para.ns:, 1]
V1_fft_stiff = np.fft.rfft(V1_stiff) / len(V1_stiff)

fig, ax = plt.subplots(2, 1, tight_layout=True)
ax1, ax2 = ax
ax1.plot(1e9 * t, 1e6 * V1[-para.ns:], label='nonstiff')
ax1.plot(1e9 * t, 1e6 * V1_stiff[-para.ns:], label='stiff')
ax2.semilogy(1e-9 * freqs, np.abs(V1_fft), label='nonstiff')
ax2.semilogy(1e-9 * freqs, np.abs(V1_fft_stiff), label='stiff')
ax1.legend()
ax2.legend()
ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Cavity V1 [uV]")
ax2.set_xlabel("Frequency [GHz]")
ax2.set_ylabel("Cavity V1 [V]")
fig.show()
