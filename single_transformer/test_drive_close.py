from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, Boltzmann
from scipy.signal import decimate

import simulator as sim

_wc = 2. * np.pi * 6e9
_kappa = 2. * np.pi * 200e3
_Qb = 1e7
_chi = _kappa / 1.
_Ql = _wc / _kappa
# _wc = 2. * np.pi * 6.0296e9
# _kappa = 2. * np.pi * 517e3
# _Qb = 1e7
# _chi = -2. * np.pi * 143e3
# _Ql = _wc / _kappa
_AMP = 1e-6  # V, will be adjusted later to reach max_ph
max_ph = 227
# max_ph = None
max_int_ph = 1.95e-4
detuning = 10.
double = True
# method = "sine"
# method = "triangle"
# method = "smart"
# method = "stupid"
method = "crazy"
nr_freqs = 4

res, para_g, para_e = sim.SimulationParameters.from_measurement(
    _wc, _chi, _Qb, _Ql)
# res, para_g = sim.SimulationParameters.from_measurement_single(_wc - _chi, _Qb, _Ql)
# res, para_e = sim.SimulationParameters.from_measurement_single(_wc + _chi, _Qb, _Ql)
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
T0 = 0.  # Tph  # K

_fc = w_c / (2. * np.pi)
_df = 2. * np.abs(detuning * kappa) / (2. * np.pi)
fc, df = para_g.tune(_fc, _df, priority='f')
para_g.set_df(df)
para_e.set_df(df)

# # reach equilibrium
# Nrelax = int(round(2. * np.pi * df / kappa))
# para_g.set_Nbeats(3 * Nrelax)
# para_e.set_Nbeats(3 * Nrelax)
# para_g.set_noise_T(T1=Tph, T0=T0)
# para_e.set_noise_T(T1=Tph, T0=T0)
# para_g.set_drive_none()
# para_e.set_drive_none()
# para_g.simulate(print_time=True)
# para_e.simulate(print_time=True)

# readout run
para_g.set_Nbeats(10)
para_e.set_Nbeats(10)
para_g.set_noise_T(T1=Tph, T0=T0)
para_e.set_noise_T(T1=Tph, T0=T0)

t = para_g.get_time_arr()
t_drive = para_g.get_drive_time_arr()


def demodulate(signal, nc, bw=None):
    if bw is None:
        bw = nc
    ns = len(signal)
    s_fft = np.fft.rfft(signal) / ns
    e_fft = np.zeros(bw, dtype=np.complex128)
    karray = np.arange(nc - bw // 2, nc + bw // 2 + 1)
    e_fft[karray - nc] = s_fft[karray]
    envelope = np.fft.ifft(e_fft) * bw
    return envelope


def demodulate_time(t, bw):
    t0 = t[0]
    dt = t[1] - t[0]
    t1 = t[-1] + dt
    t_e = np.linspace(t0, t1, bw, endpoint=False)
    return t_e


for ii in range(3):
    AMP = _AMP
    carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
    window = np.zeros_like(t_drive)
    idx = np.s_[0 * para_g.ns:2 * para_g.ns]

    if method == "triangle":
        x = df
        window[:para_g.ns] = np.linspace(0, 1, para_g.ns, endpoint=False)
        window[para_g.ns:2 * para_g.ns] = np.linspace(
            1, 0, para_g.ns, endpoint=False)
        carrier1 = 0.5 * AMP * np.cos(2. * np.pi * (fc - x) * t_drive)
        carrier2 = 0.5 * AMP * np.cos(2. * np.pi * (fc + x) * t_drive + np.pi)
        carrier = carrier1 + carrier2
        drive = window * carrier

    elif method == "sine":
        if double:
            w_m = 2 * np.pi * df / 2
        else:
            w_m = 2 * np.pi * df / 4
        window[idx] = np.sin(w_m * t_drive[idx])**(nr_freqs - 1)
        drive = window * carrier

    elif method == "smart":
        wc = 2 * np.pi * fc
        fm = df / 2
        wm = 2 * np.pi * fm
        # carrier = np.cos(wc * t_drive)
        mask0 = np.zeros_like(t_drive)
        T0, T1 = 0., 0.5 / fm
        mask0[np.logical_and(t_drive >= T0, t_drive < T1)] = 1.
        s0 = np.sin(wm * t_drive)**3 * carrier
        s0 *= mask0
        s0_fft = np.fft.rfft(s0)

        fmz = 1 * fm
        wmz = 2 * np.pi * fmz
        T2 = T1 + 0.5 / fmz
        maskz = np.zeros_like(t_drive)
        maskz[np.logical_and(t_drive >= T1, t_drive < T2)] = 1.

        # s2 = np.sin(wmz * t_drive)**2 * carrier
        # s2 = carrier
        # s2 *= maskz
        # s2_fft = np.fft.rfft(s2)
        # idx_c = int(round(fc / df * para_g.Nbeats))
        # Asmart = - s1_fft[idx_c] / s2_fft[idx_c]
        # s0_fft = s1_fft + Asmart * s2_fft

        carrier1 = np.cos(2. * np.pi * fc * t_drive)
        s1 = np.sin(wmz * t_drive)**1 * carrier1
        s1 *= maskz
        s1_fft = np.fft.rfft(s1)

        fg = w_g / 2 / np.pi
        carrier2 = np.cos(2. * np.pi * fg * t_drive)
        s2 = np.sin(wmz * t_drive)**1 * carrier2
        s2 *= maskz
        s2_fft = np.fft.rfft(s2)

        fe = w_e / 2 / np.pi
        carrier3 = np.cos(2. * np.pi * fe * t_drive)
        s3 = np.sin(wmz * t_drive)**1 * carrier3
        s3 *= maskz
        s3_fft = np.fft.rfft(s3)

        # idx_g = int(round(fg / df * para_g.Nbeats))
        # idx_e = int(round(fe / df * para_g.Nbeats))
        # s1g, s2g, s3g = s1_fft[idx_g], s2_fft[idx_g], s3_fft[idx_g]
        # s1e, s2e, s3e = s1_fft[idx_e], s2_fft[idx_e], s3_fft[idx_e]
        idx_c = int(round(fc / df * para_g.Nbeats))

        s0, s1, s2 = s0_fft[idx_c], np.gradient(s0_fft)[idx_c], np.gradient(
            np.gradient(s0_fft))[idx_c]
        p10, p11, p12 = s1_fft[idx_c], np.gradient(s1_fft)[idx_c], np.gradient(
            np.gradient(s1_fft))[idx_c]
        p20, p21, p22 = s2_fft[idx_c], np.gradient(s2_fft)[idx_c], np.gradient(
            np.gradient(s2_fft))[idx_c]
        p30, p31, p32 = s3_fft[idx_c], np.gradient(s3_fft)[idx_c], np.gradient(
            np.gradient(s3_fft))[idx_c]
        a1 = (-s0 * (p21 * p32 - p22 * p31) + s1 *
              (p20 * p32 - p22 * p30) - s2 * (p20 * p31 - p21 * p30)) / (
                  p10 * p21 * p32 - p10 * p22 * p31 - p11 * p20 * p32 +
                  p11 * p22 * p30 + p12 * p20 * p31 - p12 * p21 * p30)
        a2 = (s0 * (p11 * p32 - p12 * p31) - s1 *
              (p10 * p32 - p12 * p30) + s2 * (p10 * p31 - p11 * p30)) / (
                  p10 * p21 * p32 - p10 * p22 * p31 - p11 * p20 * p32 +
                  p11 * p22 * p30 + p12 * p20 * p31 - p12 * p21 * p30)
        a3 = (-s0 * (p11 * p22 - p12 * p21) + s1 *
              (p10 * p22 - p12 * p20) - s2 * (p10 * p21 - p11 * p20)) / (
                  p10 * p21 * p32 - p10 * p22 * p31 - p11 * p20 * p32 +
                  p11 * p22 * p30 + p12 * p20 * p31 - p12 * p21 * p30)

        final_fft = s0_fft + a1 * s1_fft + a2 * s2_fft + a3 * s3_fft
        final_fft = s0_fft

        drive = np.fft.irfft(final_fft, n=len(carrier))

    elif method == 'stupid':
        w_m = 2 * np.pi * df / 4
        window[idx] = np.sin(w_m * t_drive[idx])**(nr_freqs - 1)
        drive = window * carrier
        drive_fft = np.fft.rfft(drive)
        idx_c = int(round(fc / df * para_g.Nbeats))
        xx = np.arange(len(drive_fft))
        mask = 1 - np.exp(-0.5 * ((xx - idx_c) / 3)**2)
        drive_fft *= mask
        drive = np.fft.irfft(drive_fft, n=len(drive))

    elif method == "crazy":
        x = df / 2
        wm = 2 * np.pi * x
        carrier1 = 0.5 * AMP * np.cos(2. * np.pi * (fc - x) * t_drive)
        carrier2 = 0.5 * AMP * np.cos(2. * np.pi * (fc + x) * t_drive)
        drive1 = np.sin(wm * t_drive)**2 * carrier1
        drive2 = np.sin(wm * t_drive)**2 * carrier2
        window = np.zeros_like(t_drive)
        window[:2 * para_g.ns] = 1.
        drive = window * (drive1 + drive2)

    else:
        raise NotImplementedError

    para_g.set_drive_V(drive)
    para_e.set_drive_V(drive)

    Vg = drive[:-1]

    sol_g = para_g.simulate(print_time=True)  # , continue_run=True)
    I0_g = sol_g[:, 0]
    P1_g = sol_g[:, 1]
    V1_g = sol_g[:, 2]
    V2_g = para_g.calculate_V2(I0_g)
    sol_e = para_e.simulate(print_time=True)  # , continue_run=True)
    I0_e = sol_e[:, 0]
    P1_e = sol_e[:, 1]
    V1_e = sol_e[:, 2]
    V2_e = para_e.calculate_V2(I0_e)

    nc = int(round(fc / df))
    Nbeats = para_g.Nbeats
    t_envelope = demodulate_time(t, nc * Nbeats)
    Vg_envelope = demodulate(Vg, nc * Nbeats)
    I0_g_envelope = demodulate(I0_g, nc * Nbeats)
    P1_g_envelope = demodulate(P1_g, nc * Nbeats)
    V1_g_envelope = demodulate(V1_g, nc * Nbeats)
    V2_g_envelope = demodulate(V2_g, nc * Nbeats)
    I0_e_envelope = demodulate(I0_e, nc * Nbeats)
    P1_e_envelope = demodulate(P1_e, nc * Nbeats)
    V1_e_envelope = demodulate(V1_e, nc * Nbeats)
    V2_e_envelope = demodulate(V2_e, nc * Nbeats)

    nr_ph_g = 0.5 * 2. * np.abs(P1_g_envelope)**2 / para_g.L1 / (hbar * w_g)
    nr_ph_e = 0.5 * 2. * np.abs(P1_e_envelope)**2 / para_e.L1 / (hbar * w_e)
    idx_check = np.argmin(np.abs(t_envelope - t[idx.stop]))
    left_ph_g = nr_ph_g[idx_check]
    left_ph_e = nr_ph_e[idx_check]
    int_ph_g = np.trapz(nr_ph_g[:idx_check],
                        t_envelope[:idx_check]) + left_ph_g / kappa_g
    int_ph_e = np.trapz(nr_ph_e[:idx_check],
                        t_envelope[:idx_check]) + left_ph_e / kappa_e

    if max_ph is None:
        _AMP = AMP * np.sqrt(max_int_ph / max(int_ph_g, int_ph_e))
    else:
        _AMP = AMP * np.sqrt(max_ph / max(nr_ph_g.max(), nr_ph_e.max()))

print("Drive amplitude: {:.4g} V".format(AMP))
print("Max photons: {:.3g}".format(max(nr_ph_g.max(), nr_ph_e.max())))
print("Integral: {:.3g} ph s".format(max(int_ph_g, int_ph_e)))
print("Photons left: {:.2g}".format(max(left_ph_g, left_ph_e)))
time_to_half_g = 1 / kappa_g * np.log(2 * left_ph_g)
time_to_half_e = 1 / kappa_e * np.log(2 * left_ph_e)
print("Time to half photon: {:.2g}".format(
    max(time_to_half_g, time_to_half_e)))

distance = np.abs(V2_e_envelope - V2_g_envelope)

fig1, ax1 = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax11, ax12, ax13 = ax1

ax11.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(Vg_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(V2_g_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(V2_e_envelope))
ax12.plot(1e6 * t_envelope, 1e6 * 2 * distance)
ax13.axvline(1e6 * t_envelope[idx_check], ls='--', color='tab:gray')
ax13.plot(1e6 * t_envelope, nr_ph_g)
ax13.plot(1e6 * t_envelope, nr_ph_e)

ax11.set_title("Amplitude")
ax13.set_xlabel(r"Time [$\mathrm{\mu s}$]")
fig1.show()

fig2, ax2 = plt.subplots(3, 1, sharex=True, tight_layout=True)
ax21, ax22, ax23 = ax2

ax21.plot(1e6 * t_envelope, np.angle(Vg_envelope))
ax22.plot(1e6 * t_envelope, np.angle(V2_g_envelope))
ax22.plot(1e6 * t_envelope, np.angle(V2_e_envelope))
ax23.axvline(1e6 * t_envelope[idx_check], ls='--', color='tab:gray')
ax23.plot(1e6 * t_envelope, np.angle(V1_g_envelope))
ax23.plot(1e6 * t_envelope, np.angle(V1_e_envelope))

ax21.set_title("Phase")
ax23.set_xlabel(r"Time [$\mathrm{\mu s}$]")
fig2.show()
