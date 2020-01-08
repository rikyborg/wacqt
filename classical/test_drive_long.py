from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, Boltzmann

from utils import demodulate, demodulate_time

from simulators import sim_transformer as sim
# from simulators import sim_notch as sim
# from simulators import sim_reflection as sim
# from simulators import sim_transmission as sim

# _wr = 2. * np.pi * 6_029_359_800.
# wb = 2 * np.pi * 6_028_233_000.
# _Ql = 6_500.
# _Qc = 6_670.
# _Qb = 1 / (1 / _Ql - 1 / _Qc)
# _kappa = wb / _Qc
# _chi = -0.286 * _kappa

_wr = 2. * np.pi * 6e9
wb = 2 * np.pi * 6e9
_Ql = 6e3
_Qb = 600e3
_kappa = wb / _Ql
_chi = -0.1 * _kappa

_AMP = 1e-6  # V, will be adjusted later to reach max_ph
max_ph = 10.
nr_freqs = 2
factor = 0

res, para_g, para_e = sim.SimulationParameters.from_measurement(
    _wr, _chi, _Qb, _Ql)
assert res.success

w_g, Ql_g = para_g.calculate_resonance()
w_e, Ql_e = para_e.calculate_resonance()
Qb_g = para_g.R1 * np.sqrt(para_g.C1 / para_g.L1)
Qb_e = para_e.R1 * np.sqrt(para_e.C1 / para_e.L1)
Qc_g = 1 / (1 / Ql_g - 1 / Qb_g)
Qc_e = 1 / (1 / Ql_e - 1 / Qb_e)
chi = 0.5 * (w_e - w_g)
wr = 0.5 * (w_g + w_e)
kappa_g = w_g / Qc_g
kappa_e = w_e / Qc_e
kappa = 0.5 * (kappa_g + kappa_e)

Eph = hbar * wr
Tph = 0.  # 0.5 * Eph / Boltzmann

_fc = wr / (2. * np.pi)
_df = 1 * kappa / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)

# readout run
para_g.set_Nbeats(4)
para_e.set_Nbeats(4)
para_g.set_noise_T(T1=Tph)
para_e.set_noise_T(T1=Tph)

t = para_g.get_time_arr()
# t_drive = para_g.get_drive_time_arr()
t_drive = para_g.get_time_arr()

# first pulse
idx_start_p1, idx_stop_p1 = 0, para_g.ns
carrier = np.cos(2. * np.pi * fc * t_drive)
window_p1 = np.zeros_like(t_drive)
idx_p1 = np.s_[idx_start_p1:idx_stop_p1]
window_p1[idx_p1] = np.sin(2. * np.pi * 0.5 * df * t_drive[idx_p1])**(nr_freqs - 1)
_drive_p1 = window_p1 * carrier

if factor:
    # second pulse
    idx_start_p2 = idx_stop_p1
    idx_stop_p2 = idx_start_p2 + para_g.ns // factor
    t0 = t_drive[idx_start_p2]
    window_p2 = np.zeros_like(t_drive)
    idx_p2 = np.s_[idx_start_p2:idx_stop_p2]
    window_p2[idx_p2] = np.sin(2. * np.pi * 0.5 * factor * df * (t_drive[idx_p2] - t0))**(nr_freqs - 1)
    _drive_p2 = window_p2 * carrier

    # adjust second pulse
    p1_fft = np.fft.rfft(_drive_p1)
    p2_fft = np.fft.rfft(_drive_p2)
    idx_fft = int(round(fc / df)) * para_g.Nbeats
    p1_amp = p1_fft[idx_fft]
    p2_amp = p2_fft[idx_fft]
    new_p2_amp = -p1_amp
    new_p2_fft = p2_fft / p2_amp * new_p2_amp
    _new_drive_p2 = np.fft.irfft(new_p2_fft, n=len(_drive_p2))

    # # 0.25
    # len_p2 = idx_stop_p2 - idx_start_p2
    # new_len_p2 = int(round(len_p2 / 15.04))
    # new_idx_stop_p2 = idx_start_p2 + new_len_p2
    # _new_drive_p2[new_idx_stop_p2:] = 0.

    # # 0.5
    # len_p2 = idx_stop_p2 - idx_start_p2
    # new_len_p2 = int(round(len_p2 / 7.354))
    # new_idx_stop_p2 = idx_start_p2 + new_len_p2
    # _new_drive_p2[new_idx_stop_p2:] = 0.

    # # 1
    # len_p2 = idx_stop_p2 - idx_start_p2
    # # new_len_p2 = int(round(len_p2 / 3.759))
    # new_len_p2 = int(round(len_p2 / 3.265))
    # new_idx_stop_p2 = idx_start_p2 + new_len_p2
    # _new_drive_p2[new_idx_stop_p2:] = 0.
    _new_drive_p2 /= 3.265

    # # 2
    # len_p2 = idx_stop_p2 - idx_start_p2
    # new_len_p2 = int(round(len_p2 / 2.220))
    # new_idx_stop_p2 = idx_start_p2 + new_len_p2
    # _new_drive_p2[new_idx_stop_p2:] = 0.

    _drive = _drive_p1 + _new_drive_p2
else:
    _drive = _drive_p1
    idx_p2 = idx_p1

_drive = np.concatenate((_drive, [0.]))

for ii in range(3):
    AMP = _AMP
    drive = AMP * _drive

    para_g.set_drive_V(drive)
    para_e.set_drive_V(drive)

    Vg = drive[:-1]

    sol_g = para_g.simulate(print_time=True)  # , continue_run=True)
    P1_g = sol_g[:, 1]
    sol_e = para_e.simulate(print_time=True)  # , continue_run=True)
    P1_e = sol_e[:, 1]

    nc = int(round(fc / df))
    Nbeats = para_g.Nbeats
    t_envelope = demodulate_time(t, nc * Nbeats)
    Vg_envelope = demodulate(Vg, nc * Nbeats)
    P1_g_envelope = demodulate(P1_g, nc * Nbeats)
    P1_e_envelope = demodulate(P1_e, nc * Nbeats)

    nr_ph_g = 0.5 * 2. * np.abs(P1_g_envelope)**2 / para_g.L1 / (hbar * w_g)
    nr_ph_e = 0.5 * 2. * np.abs(P1_e_envelope)**2 / para_e.L1 / (hbar * w_e)

    _AMP = AMP * np.sqrt(max_ph / max(nr_ph_g.max(), nr_ph_e.max()))

Vout_g = para_g.calculate_Vout(sol_g)
Vout_g_envelope = demodulate(Vout_g, nc * Nbeats)
Vout_e = para_e.calculate_Vout(sol_e)
Vout_e_envelope = demodulate(Vout_e, nc * Nbeats)

idx_check = np.argmin(np.abs(t_envelope - t[idx_p1.stop]))
left_ph_g = nr_ph_g[idx_check]
left_ph_e = nr_ph_e[idx_check]
print("Drive amplitude: {:.4g} V".format(AMP))
print("Max photons: {:.3g}".format(max(nr_ph_g.max(), nr_ph_e.max())))
print("Photons left: {:.2g}".format(max(left_ph_g, left_ph_e)))
time_to_half_g = 1 / kappa_g * np.log(2 * left_ph_g)
time_to_half_e = 1 / kappa_e * np.log(2 * left_ph_e)
print("Time to half photon: {:.2g}".format(
    max(time_to_half_g, time_to_half_e)))

fig1, ax1 = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax11, ax12, ax13 = ax1

for _ax in ax1:
    idx_1 = np.argmin(np.abs(t_envelope - t[idx_p1.stop]))
    idx_2 = np.argmin(np.abs(t_envelope - t[idx_p2.stop]))
    _ax.axvline(1e6 * t_envelope[idx_1], ls='--', color='tab:gray')
    _ax.axvline(1e6 * t_envelope[idx_2], ls='--', color='tab:gray')

ax11.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vg_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vout_g_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vout_e_envelope))
# ax13.axvspan(1e6 * t_envelope[Npoints // 2], 1e6 * t_envelope[3 * Npoints // 4], color='tab:gray', alpha=0.5)
ax13.plot(1e6 * t_envelope, nr_ph_g)
ax13.plot(1e6 * t_envelope, nr_ph_e)

ax11.set_title("Amplitude")
ax13.set_xlabel(r"Time [$\mathrm{\mu s}$]")

ax11.set_ylabel(r"$\left|V_\mathrm{G}\right|$ [$\mathrm{\mu V}$]")
ax12.set_ylabel(r"$\left|V_\mathrm{OUT}\right|$ [$\mathrm{\mu V}$]")
ax13.set_ylabel(r"$\left<n_\mathrm{PH}\right>$")

fig1.show()

fig2, ax2 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax21, ax22 = ax2

ax21.plot(1e6 * t_envelope, np.angle(Vg_envelope))
ax22.plot(1e6 * t_envelope, np.angle(Vout_g_envelope))
ax22.plot(1e6 * t_envelope, np.angle(Vout_e_envelope))
# ax23.axvline(1e6 * t_envelope[idx_check], ls='--', color='tab:gray')
# ax23.plot(1e6 * t_envelope, np.angle(V1_g_envelope))
# ax23.plot(1e6 * t_envelope, np.angle(V1_e_envelope))

ax21.set_title("Phase")
ax22.set_xlabel(r"Time [$\mathrm{\mu s}$]")

ax21.set_ylabel(
    r"$\operatorname{Arg}\left(V_\mathrm{G}\right)$ [$\mathrm{rad}$]")
ax22.set_ylabel(
    r"$\operatorname{Arg}\left(V_\mathrm{OUT}\right)$ [$\mathrm{rad}$]")

fig2.show()
