from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np

import simulator as sim

para = sim.SimulationParameters(
    Cl=1e-15, Cr=1e-12,
    R1=3162., L1=1e-9, C1=1e-12,
    R2=3162., L2=2e-9, C2=2e-12,
    w_arr=[1e9, 2e9], A_arr=[0.5, 0.5], P_arr=[0., np.pi],
)
Nr = 5  # nr windows to reach steady state (throw away)
Na = 4  # nr windows to FFT
para.set_Nbeats(Nr + Na)
para.set_duffing(1e23)
# para.set_noise_T(300.)

df = 50e6  # Hz
fc = 4.04e9  # Hz
f1 = fc - df / 2.
f1_, df_ = para.tune(f1, df, prio_f=True)
para.set_df(df_)
para.set_drive_frequencies([f1_, f1_ + df_, ])

sol = para.simulate()

V1 = sol[-Na * para.ns:, 1]
V1_fft = np.fft.rfft(V1) / len(V1)
freqs = np.fft.rfftfreq(Na * para.ns, para.dt)

fig, ax = plt.subplots(tight_layout=True)
ax.semilogy(freqs, np.abs(V1_fft))
fig.show()
