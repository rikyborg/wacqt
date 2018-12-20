from __future__ import division, print_function

import time

from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
from scipy.optimize import least_squares

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
df = 2e6  # Hz

# para = sim.SimulationParameters(
#     Cl=1e-11,
#     R1=100., L1=1e-9, C1=1e-9,
#     fs=30e9,
# )
# df = 1e6

AMP = 0.5e-6  # V
PHASE = 0.  # rad
fstart = para.f01_d * (1. - 10. / para.Q1_d)
fstop = para.f01_d * (1. + 10. / para.Q1_d)

# f_array1 = np.linspace(fstart, fstop, 500, endpoint=False)
# f_array2 = np.linspace(fstop, fstart, 500, endpoint=False)
# f_array = np.concatenate((f_array1, f_array2))

f_array = np.linspace(fstart, fstop, 501)

PHI0 = 2.067833831e-15  # Wb, magnetic flux quantum
fake_PHI0 = PHI0 * 20.  # junctions in series
# para.set_duffing((2. * np.pi / fake_PHI0)**2 / 6)
# para.set_josephson(PHI0=fake_PHI0)

actual_f_array = np.zeros_like(f_array)
resp0_array = np.zeros_like(f_array, dtype=np.complex128)
resp1_array = np.zeros_like(f_array, dtype=np.complex128)
reflected_array = np.zeros_like(f_array, dtype=np.complex128)

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
    fd_, df_ = para.tune(fd, df, priority='f')
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
    reflected_array[ii] = Vr_fft[nd_]
t_end = time.time()
t_tot = t_end - t_start
print("Total run took {:s}.".format(sim.format_sec(t_tot)))

resp0_array *= 2. / (AMP * np.exp(1j * PHASE))
resp1_array *= 2. / (AMP * np.exp(1j * PHASE))

G0 = para.tf0(f_array)
G1 = para.tf1(f_array)


def lorentzian(f, f0, Q, A, P):
    return A * np.exp(1j * P) / (1. - f**2 / f0**2 + 1j * f / f0 / Q)


def erf(p, f, r):
    f0, Q, A, P = p
    e = lorentzian(f, f0, Q, A, P) - r
    return np.concatenate((e.real, e.imag))


# def lorentzian0(f, f0, Q, A):
#     w = f0 / Q
#     x = (f - f0) / (w / 2)
#     return np.sqrt(1. + A / (1. + x**2))
#     # return np.sqrt(1. + A * np.abs(lorentzian(f, f0, Q, A=1., P=0.))**2)


# def erf0(p, f, r):
#     f0, Q, A = p
#     e = lorentzian0(f, f0, Q, A) - np.abs(r)
#     return e


def on_mouse_in_canvas(event):
    if fig.canvas.manager.toolbar._active is None:
        # hspan_sel0.visible = True
        hspan_sel1.visible = True
    else:
        # hspan_sel0.visible = False
        hspan_sel1.visible = False


# def fit_lorentzian0(fmin, fmax):
#     idx = np.logical_and(f_array > 1e9 * fmin, f_array < 1e9 * fmax)
#     f_fit = f_array[idx]
#     r_fit = resp0_array[idx]
#     if len(f_fit) < 4:
#         return
#     res = least_squares(
#         erf0, [para.f01_d, para.Q1_d, -0.5],
#         # method='lm',
#         args=(f_fit, r_fit),
#     )
#     lf0a.set_data(1e-9 * f_array, 20. * np.log10(lorentzian0(f_array, *res.x)))
#     # lf0p.set_data(1e-9 * f_array, np.angle(1 + lorentzian0(f_array, *pfit)))
#     print("f0 = {:.3g} GHz, Q = {:.3g}".format(res.x[0] / 1e9, res.x[1]))
#     print(res.x)


def fit_lorentzian1(fmin, fmax):
    idx = np.logical_and(f_array > 1e9 * fmin, f_array < 1e9 * fmax)
    f_fit = f_array[idx]
    r_fit = resp1_array[idx]
    if len(f_fit) < 4:
        return
    res = least_squares(
        erf, [para.f01_d, para.Q1_d, 5.5e-2, np.pi],
        # bounds=([fstart, 1., 0., -2. * np.pi], [fstop, np.inf, np.inf, 2. * np.pi]),
        method='lm',
        args=(f_fit, r_fit),
    )
    lf1a.set_data(1e-9 * f_array, 20. * np.log10(np.abs(lorentzian(f_array, *res.x))))
    lf1p.set_data(1e-9 * f_array, np.angle(lorentzian(f_array, *res.x)))
    print("f0 = {:.3g} GHz, Q = {:.3g}".format(res.x[0] / 1e9, res.x[1]))
    print(res.x)


fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
fig.canvas.mpl_connect('axes_enter_event', on_mouse_in_canvas)
ax0, ax1 = ax
ax0r, ax1r = ax0.twinx(), ax1.twinx()
axr = ax0r, ax1r
for ax_ in ax:
    ax_.spines['left'].set_color('tab:blue')
    ax_.tick_params(axis='y', which='both', colors='tab:blue')
    ax_.yaxis.label.set_color('tab:blue')
    # ticks = [1e9 * nn / para.df for nn in range(para.Nbeats + 1)]
    # ticks_minor = [1e9 * (nn + 0.5) / para.df for nn in range(para.Nbeats)]
    # ax_.set_xticks(ticks)
    # ax_.set_xticks(ticks_minor, minor=True)
    # for TT in ticks:
    #     ax_.axvline(TT, ls='--', c='tab:gray')
for ax_ in axr:
    ax_.set_ylim(-np.pi, np.pi)
    ax_.spines['right'].set_color('tab:orange')
    ax_.tick_params(axis='y', which='both', colors='tab:orange')
    ax_.yaxis.label.set_color('tab:orange')
    ax_.set_yticks([-np.pi, 0., np.pi])
    ax_.set_yticks([-np.pi / 2, np.pi / 2], minor=True)
    ax_.set_yticklabels([u'\u2212\u03c0', '0', u'\u002b\u03c0'])

# hspan_sel0 = SpanSelector(ax0r, fit_lorentzian0, 'horizontal', rectprops=dict(facecolor='tab:red', alpha=0.3))
hspan_sel1 = SpanSelector(ax1r, fit_lorentzian1, 'horizontal', rectprops=dict(facecolor='tab:red', alpha=0.3))

ax0.plot(1e-9 * f_array, 20. * np.log10(np.abs(G0)), '-', c='tab:blue')
ax0.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(resp0_array)), '.', c='tab:blue', label=r'$V_0$')
ax1.plot(1e-9 * f_array, 20. * np.log10(np.abs(G1)), '-', c='tab:blue')
ax1.plot(1e-9 * actual_f_array, 20. * np.log10(np.abs(resp1_array)), '.', c='tab:blue', label=r'$V_1$')
# lf0a, = ax0.plot([], [], '--', c='tab:red')
lf1a, = ax1.plot([], [], '--', c='tab:red')

ax0r.plot(1e-9 * f_array, np.angle(G0), '-', c='tab:orange')
ax0r.plot(1e-9 * actual_f_array, np.angle(resp0_array), '.', c='tab:orange')
ax1r.plot(1e-9 * f_array, np.angle(G1), '-', c='tab:orange')
ax1r.plot(1e-9 * actual_f_array, np.angle(resp1_array), '.', c='tab:orange')
# lf0p, = ax0r.plot([], [], '--', c='tab:red')
lf1p, = ax1r.plot([], [], '--', c='tab:red')

for ax_ in ax:
    ax_.set_ylabel(r"Amplitude [dB]")
    ax_.legend()
for ax_ in axr:
    ax_.set_ylabel(r"Phase [rad]")
ax1.set_xlabel(r"Frequency [$\mathrm{GHz}$]")
fig.show()
