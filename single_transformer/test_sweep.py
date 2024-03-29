from __future__ import absolute_import, division, print_function

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

# para = sim.SimulationParameters(
#     Lg=1e-12,
#     R1=1000., L1=1e-9 / (2. * np.pi), C1=1e-9 / (2. * np.pi),
#     R0=50., R2=50.,
#     fs=3e9,
# )
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
respI0_array = np.zeros_like(f_array, dtype=np.complex128)
respP1_array = np.zeros_like(f_array, dtype=np.complex128)
respV1_array = np.zeros_like(f_array, dtype=np.complex128)
respV2_array = np.zeros_like(f_array, dtype=np.complex128)
respV0_array = np.zeros_like(f_array, dtype=np.complex128)

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

    I0 = sol[-para.ns:, 0]
    P1 = sol[-para.ns:, 1]
    V1 = sol[-para.ns:, 2]
    Vg = para.get_drive_V()[-para.ns - 1:-1]
    V0 = para.calculate_V0(I0, Vg)
    V2 = para.calculate_V2(I0)

    I0_fft = np.fft.rfft(I0) / para.ns
    P1_fft = np.fft.rfft(P1) / para.ns
    V1_fft = np.fft.rfft(V1) / para.ns
    V2_fft = np.fft.rfft(V2) / para.ns
    V0_fft = np.fft.rfft(V0) / para.ns
    actual_f_array[ii] = fd_
    respI0_array[ii] = I0_fft[nd_]
    respP1_array[ii] = P1_fft[nd_]
    respV1_array[ii] = V1_fft[nd_]
    respV2_array[ii] = V2_fft[nd_]
    respV0_array[ii] = V0_fft[nd_]
t_end = time.time()
t_tot = t_end - t_start
print("Total run took {:s}.".format(sim.format_sec(t_tot)))

respV0_array *= 2. / (AMP * np.exp(1j * PHASE))
respV1_array *= 2. / (AMP * np.exp(1j * PHASE))
respV2_array *= 2. / (AMP * np.exp(1j * PHASE))

f_plot = np.linspace(fstart, fstop, 20001)
G0 = para.tf0(f_plot)
G1 = para.tf1(f_plot)
G2 = para.tf2(f_plot)


fig, ax = plt.subplots(2, 3, sharex=True, tight_layout=True)
ax1, ax2 = ax
ax11, ax12, ax13 = ax1
ax21, ax22, ax23 = ax2

for _ax in ax.flatten():
    _ax.axvline(1e-9 * f0, ls='--', c='tab:gray')
    _ax.axvline(1e-9 * (f0 - bw / 2), ls='--', c='tab:gray')
    _ax.axvline(1e-9 * (f0 + bw / 2), ls='--', c='tab:gray')

ax11.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(respV0_array)), '.', c='tab:blue')
ax11.plot(1e-9 * f_plot, 20. * np.log10(np.abs(G0)), '--', c='tab:green')
ax21.plot(1e-9 * actual_f_array, np.angle(respV0_array), '.', c='tab:orange')
ax21.plot(1e-9 * f_plot, np.angle(G0), '--', c='tab:red')
ax11.set_title('Transmission line')

ax12.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(respV1_array)), '.', c='tab:blue')
ax12.plot(1e-9 * f_plot, 20. * np.log10(np.abs(G1)), '--', c='tab:green')
ax22.plot(1e-9 * actual_f_array, np.angle(respV1_array), '.', c='tab:orange')
ax22.plot(1e-9 * f_plot, np.angle(G1), '--', c='tab:red')
ax12.set_title('Oscillator')

ax13.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(respV2_array)), '.', c='tab:blue')
ax13.plot(1e-9 * f_plot, 20. * np.log10(np.abs(G2)), '--', c='tab:green')
ax23.plot(1e-9 * actual_f_array, np.angle(respV2_array), '.', c='tab:orange')
ax23.plot(1e-9 * f_plot, np.angle(G2), '--', c='tab:red')
ax13.set_title('Output port')

fig.show()
