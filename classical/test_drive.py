from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, Boltzmann

from utils import demodulate, demodulate_time

# from simulators import sim_transformer as sim
from simulators import sim_notch as sim
# from simulators import sim_reflection as sim
# from simulators import sim_transmission as sim

_wc = 2. * np.pi * 6e9
_chi = 2. * np.pi * 2e6
_Qb = 1e7
# _kappa = 2. * np.pi * 37.5e6 / 1e2
_kappa = _chi / 10.
_Ql = _wc / _kappa
# _Qb = 100 * _Ql

_AMP = 1e-6  # V, will be adjusted later to reach max_ph
max_ph = 10.
nr_freqs = 3
double = True
method = "sine"
# method = "triangle"
# method = "triangle_new"

res, para_g, para_e = sim.SimulationParameters.from_measurement(
    _wc, _chi, _Qb, _Ql)
assert res.success

w_g, Q_g = para_g.calculate_resonance()
w_e, Q_e = para_e.calculate_resonance()
chi = 0.5 * (w_e - w_g)
w_c = 0.5 * (w_g + w_e)
kappa_g = w_g / Q_g
kappa_e = w_e / Q_e
kappa = 0.5 * (kappa_g + kappa_e)

Eph = hbar * w_c
Tph = 0.  # 0.5 * Eph / Boltzmann

_fc = w_c / (2. * np.pi)
if method == "triangle_new":
    _df = np.abs(chi) / (2. * np.pi)
else:
    _df = 2. * np.abs(chi) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)

# readout run
if method == "triangle_new":
    para_g.set_Nbeats(2)
    para_e.set_Nbeats(2)
else:
    para_g.set_Nbeats(4)
    para_e.set_Nbeats(4)
para_g.set_noise_T(T1=Tph)
para_e.set_noise_T(T1=Tph)

t = para_g.get_time_arr()
# t_drive = para_g.get_drive_time_arr()

if method == "sine":
    carrier = np.cos(2. * np.pi * fc * t)
    window = np.zeros_like(t)
    if double:
        idx = np.s_[0 * para_g.ns:2 * para_g.ns]
    else:
        idx = np.s_[0 * para_g.ns:1 * para_g.ns]
    window[idx] = np.sin(2. * np.pi * 0.5 * df * t[idx])**(nr_freqs - 1)
    _drive = window * carrier
elif method == "triangle":
    carrier = np.cos(2. * np.pi * fc * t)
    window = np.zeros_like(t)
    window[:2 * para_g.ns] = np.linspace(0, 1, 2 * para_g.ns, endpoint=False)
    window[2 * para_g.ns:4 * para_g.ns] = np.linspace(1, 0, 2 * para_g.ns, endpoint=False)
    window *= np.sin(2. * np.pi * 0.5 * df * t)**(nr_freqs - 1)
    _drive = window * carrier
    idx = np.s_[0 * para_g.ns:4 * para_g.ns]
elif method == "triangle_new":
    carrier = np.cos(2. * np.pi * fc * t)
    window = np.zeros_like(t)
    window[:para_g.ns // 2] = np.linspace(0, 1, para_g.ns // 2, endpoint=False)
    window[para_g.ns // 2:para_g.ns] = np.linspace(1, 0, para_g.ns // 2, endpoint=False)
    window *= np.sin(2. * np.pi * 3 * df * t)**(nr_freqs - 2)
    _drive = window * carrier
    idx = np.s_[0 * para_g.ns:1 * para_g.ns]
else:
    raise NotImplementedError
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

idx_check = np.argmin(np.abs(t_envelope - t[idx.stop]))
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

ax11.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vg_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vout_g_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vout_e_envelope))
# ax13.axvspan(1e6 * t_envelope[Npoints // 2], 1e6 * t_envelope[3 * Npoints // 4], color='tab:gray', alpha=0.5)
ax13.axvline(1e6 * t_envelope[idx_check], ls='--', color='tab:gray')
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
