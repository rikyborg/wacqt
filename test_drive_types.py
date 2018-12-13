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
)
para.set_Nbeats(9)
para.set_duffing(1e25)
# para.set_noise_T(300.)

df_ = 50e6  # Hz
fc_ = 4.04e9  # Hz
f1_ = fc_ - df_ / 2.
f1, df = para.tune(f1_, df_, priority='f', regular=True)
f_arr = np.array([f1, f1 + df])
A_arr = np.array([0.5, 0.5])
P_arr = np.array([0., np.pi])

para.set_df(df)

print("*** lockin")
para.set_drive_lockin(f_arr, A_arr, P_arr)
sol1 = para.simulate(print_time=True)

print("*** drive V")
t_drive = para.get_time_arr(extra=True)
drive = np.zeros_like(t_drive)
for ii in range(len(f_arr)):
    drive += A_arr[ii] * np.cos(2. * np.pi * f_arr[ii] * t_drive + P_arr[ii])
para.set_drive_V(drive)
sol2 = para.simulate(print_time=True)

t = para.dt * np.arange(para.ns)
V1 = sol1[-para.ns:, 2]
V2 = sol2[-para.ns:, 2]
freqs = np.fft.rfftfreq(4 * para.ns, para.dt)
Vfft1 = np.fft.rfft(sol1[-4 * para.ns:, 1])
Vfft2 = np.fft.rfft(sol2[-4 * para.ns:, 1])

fig1, ax1 = plt.subplots(2, 1, tight_layout=True)
ax11, ax12 = ax1
ax11.plot(1e9 * t, V1, label='lockin')
ax11.plot(1e9 * t, V2, label='drive V')
ax11.legend()
ax11.set_xlabel("Time [ns]")
ax11.set_ylabel("Cavity voltage V1 [V]")
ax12.semilogy(1e-9 * freqs, np.abs(Vfft1), label='lockin')
ax12.semilogy(1e-9 * freqs, np.abs(Vfft2), label='drive V')
ax12.legend()
ax12.set_xlabel("Frequency [GHz]")
ax12.set_ylabel("Cavity voltage V1 [V]")
fig1.show()
