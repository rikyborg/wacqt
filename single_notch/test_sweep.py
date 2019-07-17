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
_, para = sim.SimulationParameters.from_measurement_single(2. * np.pi * 1e9, 1e3, 1e2)
df = 1e6  # Hz

w0, Q = para.calculate_resonance()
f0 = w0 / (2. * np.pi)

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
resp0_array = np.zeros_like(f_array, dtype=np.complex128)
resp1_array = np.zeros_like(f_array, dtype=np.complex128)

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

    V0 = sol[-para.ns:, 0]
    V0_fft = np.fft.rfft(V0) / para.ns
    V1 = sol[-para.ns:, 2]
    V1_fft = np.fft.rfft(V1) / para.ns
    Vg = para.get_drive_V()[-para.ns - 1:-1]
    Vr = V0 - 0.5 * Vg
    Vr_fft = np.fft.rfft(Vr) / para.ns
    actual_f_array[ii] = fd_
    resp0_array[ii] = V0_fft[nd_]
    resp1_array[ii] = V1_fft[nd_]
t_end = time.time()
t_tot = t_end - t_start
print("Total run took {:s}.".format(sim.format_sec(t_tot)))

resp0_array *= 2. / (AMP * np.exp(1j * PHASE))
resp1_array *= 2. / (AMP * np.exp(1j * PHASE))

G0 = para.tf0(f_array)
G1 = para.tf1(f_array)


fig, ax = plt.subplots(2, 2, sharex=True, tight_layout=True)
ax1, ax2 = ax
ax11, ax12 = ax1
ax21, ax22 = ax2

ax11.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(resp0_array * 1j * 2. * np.pi * actual_f_array)), '.', c='tab:blue')
ax11.plot(1e-9 * f_array, 20. * np.log10(np.abs(G0)), '--', c='tab:green')
ax21.plot(1e-9 * actual_f_array, np.angle(resp0_array * 1j * 2. * np.pi * actual_f_array), '.', c='tab:orange')
ax21.plot(1e-9 * f_array, np.angle(G0), '--', c='tab:red')
ax11.set_title('Input port')

ax12.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(resp1_array)), '.', c='tab:blue')
ax12.plot(1e-9 * f_array, 20. * np.log10(np.abs(G1)), '--', c='tab:green')
ax22.plot(1e-9 * actual_f_array, np.angle(resp1_array), '.', c='tab:orange')
ax22.plot(1e-9 * f_array, np.angle(G1), '--', c='tab:red')
ax12.set_title('Oscillator')

fig.show()
