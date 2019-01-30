from __future__ import division, print_function

from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann

import simulator as sim

# Change default font size for nicer plots
rcParams['figure.titlesize'] = 'large'
rcParams['axes.labelsize'] = 'large'
rcParams['axes.titlesize'] = 'large'
rcParams['legend.fontsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'

# PULSE = 'beat'
# PULSE = 'sin'
PULSE = 'sin2'
# PULSE = 'square'
# PULSE = 'bump'
# PULSE = 'gauss'


para = sim.SimulationParameters(
    Cl=1e-16, Cr=6.19e-15,
    R1=2.50e8, L1=9.94e-10, C1=3.98e-13,
    R2=6.24e8, L2=7.75e-9, C2=6.44e-14,  # ground, 01
    # R2=6.24e8, L2=8.22e-9, C2=6.44e-14,  # excited, 12
)

fc_ground = 7959147851.478515  # Hz
fc_excited = 7956492274.9227495  # Hz
# df_ = 1e6
df_ = (fc_ground - fc_excited)
fc_ = 0.5 * (fc_ground + fc_excited)
if PULSE == 'beat':
    f1_ = fc_ - df_ / 2.
    f1, df = para.tune(f1_, df_, priority='f')  # , regular=True)
    f2 = f1 + df
else:  # square or bump or gauss
    fc, df = para.tune(fc_, df_, priority='f')  # , regular=True)

AMP = 10e-6  # V

para.set_df(df)
para.set_Nbeats(4)

t = para.get_time_arr()
t_drive = para.get_drive_time_arr()

if PULSE == 'beat':
    drive = np.zeros_like(t_drive)
    0.5 * AMP * (np.cos(2. * np.pi * f1 * t_drive) + np.cos(2. * np.pi * f2 * t_drive + np.pi))
    drive[:para.ns] = 0.
    drive[-2 * para.ns:] = 0.
elif PULSE == 'sin':
    carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
    window = np.zeros_like(t_drive)
    window[para.ns:2 * para.ns] = -np.sin(2. * np.pi * 0.5 * df * t_drive[para.ns:2 * para.ns])
    drive = window * carrier
elif PULSE == 'sin2':
    carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
    window = np.zeros_like(t_drive)
    window[para.ns:3 * para.ns] = np.sin(2. * np.pi * 0.5 * df * t_drive[para.ns:3 * para.ns])**2
    drive = window * carrier
elif PULSE == 'square':
    carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
    window = np.zeros_like(t_drive)
    window[para.ns:2 * para.ns] = 1.
    drive = window * carrier
elif PULSE == 'bump':
    carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
    window = np.zeros_like(t_drive)
    x = 2. * ((t_drive[para.ns + 1:2 * para.ns - 1] - t_drive[para.ns]) / t_drive[para.ns] - 0.5)
    window[para.ns + 1:2 * para.ns - 1] = np.exp(1. - 1. / (1. - x**2))
    drive = window * carrier
elif PULSE == 'gauss':
    sigma = 1. / 1.
    carrier = AMP * np.cos(2. * np.pi * fc * t_drive)
    x = 2. * ((t_drive - t_drive[para.ns]) / t_drive[para.ns] - 0.5)
    window = np.exp(-(x / sigma)**2)
    drive = window * carrier

# PSDdrive_onesided = 4. * Boltzmann * 30e-3 * 50.  # 50 ohm at 30 mK
# PSDdrive_twosided = PSDdrive_onesided / 2.
# noisedrive_array = np.sqrt(PSDdrive_twosided) * np.sqrt(para.fs) * np.random.randn(len(drive))
# drive += noisedrive_array
para.set_drive_V(drive)

para.set_noise_T(30e-3)
para.set_josephson()
sol = para.simulate()
V1 = sol[:, 2]
V2 = sol[:, 4]

fig1 = plt.figure(figsize=(12.8, 4.8))
ax11 = fig1.add_subplot(3, 2, 1)
ax12 = fig1.add_subplot(3, 2, 3, sharex=ax11)
ax13 = fig1.add_subplot(3, 2, 5, sharex=ax12)
ax1r = fig1.add_subplot(1, 2, 2)
for ax_ in [ax11, ax12, ax13]:
    for nn in range(para.Nbeats + 1):
        TT = nn / para.df
        ax_.axvline(1e6 * TT, ls='--', c='tab:gray')
ax11.plot(1e6 * t_drive, 1e6 * drive, c='tab:blue', label=r'drive [$\mathrm{\mu V}$]')
ax12.plot(1e6 * t, 1e6 * V1, c='tab:orange', label=r'Cavity V1 [$\mathrm{\mu V}$]')
ax13.plot(1e6 * t, 1e6 * V2, c='tab:green', label=r'Qubit V2 [$\mathrm{\mu V}$]')
for ax_ in [ax11, ax12, ax13]:
    ax_.legend(loc='upper right')
ax13.set_xlabel(r"Time [$\mathrm{\mu s}$]")


def on_key_press(event):
    global offset, span, para, t
    if event.key == ' ':
        offset = para.ns
        span = para.ns
    elif event.key in ['left', 'right', 'up', 'down']:
        # step = 1
        step = para.ns // 1000
    elif event.key in ['q', 'w', 'i', 'o']:
        # step = 10
        step = para.ns // 100
    elif event.key in ['a', 's', 'j', 'k']:
        # step = 100
        step = para.ns // 10
    elif event.key in ['z', 'x', 'n', 'm']:
        # step = 1000
        step = para.ns // 1
    else:
        return

    if event.key in ['left', 'q', 'a', 'z']:
        offset -= step
        if offset < 0:
            offset = 0
    elif event.key in ['right', 'w', 's', 'x']:
        offset += step
        if offset > len(t) - span:
            offset = len(t) - span
    elif event.key in ['down', 'i', 'j', 'n']:
        span -= step
        if span < 1:
            span = 1
    elif event.key in ['up', 'o', 'k', 'm']:
        span += step
        if span > len(t) - offset:
            span = len(t) - offset

    update(offset, span)


def update(offset, span):
    global para, t, drive, V1, V2, s1, s2, s3, l1, l2, l3, fig1, freqs
    drive_fft = np.fft.rfft(drive[offset:offset + span]) / span
    V1_fft = np.fft.rfft(V1[offset:offset + span]) / span
    V2_fft = np.fft.rfft(V2[offset:offset + span]) / span

    # l1.set_ydata(np.abs(drive_fft))
    # l2.set_ydata(np.abs(V1_fft))
    # l3.set_ydata(np.abs(V2_fft))

    freqs = np.fft.rfftfreq(span, para.dt)
    l1.remove()
    l2.remove()
    l3.remove()
    l1, = ax1r.semilogy(1e-9 * freqs, np.abs(drive_fft), c='tab:blue')
    l2, = ax1r.semilogy(1e-9 * freqs, np.abs(V1_fft), c='tab:orange')
    l3, = ax1r.semilogy(1e-9 * freqs, np.abs(V2_fft), c='tab:green')

    pos = s1.get_xy().copy()
    pos[[0, 1, 4], 0] = 1e6 * t[offset]
    pos[[2, 3], 0] = 1e6 * t[offset + span - 1]
    s1.set_xy(pos)
    s2.set_xy(pos)
    s3.set_xy(pos)

    fig1.canvas.draw()


fig1.canvas.mpl_disconnect(fig1.canvas.manager.key_press_handler_id)
fig1.canvas.mpl_connect("key_press_event", on_key_press)

offset = para.ns
span = para.ns
s1 = ax11.axvspan(1e6 * t[offset], 1e6 * t[offset + span - 1], facecolor='tab:gray', alpha=0.25)
s2 = ax12.axvspan(1e6 * t[offset], 1e6 * t[offset + span - 1], facecolor='tab:gray', alpha=0.25)
s3 = ax13.axvspan(1e6 * t[offset], 1e6 * t[offset + span - 1], facecolor='tab:gray', alpha=0.25)

freqs = np.fft.rfftfreq(para.ns, para.dt)
drive_fft = np.fft.rfft(drive[offset:offset + span]) / span
V1_fft = np.fft.rfft(V1[offset:offset + span]) / span
V2_fft = np.fft.rfft(V2[offset:offset + span]) / span

l1, = ax1r.semilogy(1e-9 * freqs, np.abs(drive_fft), c='tab:blue')
l2, = ax1r.semilogy(1e-9 * freqs, np.abs(V1_fft), c='tab:orange')
l3, = ax1r.semilogy(1e-9 * freqs, np.abs(V2_fft), c='tab:green')

ax1r.yaxis.tick_right()
ax1r.set_xlabel("Frequency [GHz]")

fig1.tight_layout()
fig1.show()


Nimps = 31
Npoints = 1000
nc = int(round(fc / df))
karray = np.arange((nc - Nimps // 2) * para.Nbeats, (nc + Nimps // 2) * para.Nbeats + 1)
t_envelope = np.linspace(0., para.Nbeats / para.df, Npoints, endpoint=False)

drive_spectrum = np.fft.rfft(drive[:-1]) / len(drive[:-1])
V1_spectrum = np.fft.rfft(V1) / len(V1)
V2_spectrum = np.fft.rfft(V2) / len(V2)

drive_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V1_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)
V2_envelope_spectrum = np.zeros(Npoints, dtype=np.complex128)

drive_envelope_spectrum[karray - para.Nbeats * nc] = drive_spectrum[karray]
V1_envelope_spectrum[karray - para.Nbeats * nc] = V1_spectrum[karray]
V2_envelope_spectrum[karray - para.Nbeats * nc] = V2_spectrum[karray]

drive_envelope = np.fft.ifft(drive_envelope_spectrum) * Npoints
V1_envelope = np.fft.ifft(V1_envelope_spectrum) * Npoints
V2_envelope = np.fft.ifft(V2_envelope_spectrum) * Npoints

fig2, ax2 = plt.subplots(3, 1, tight_layout=True, sharex=True)
ax21, ax22, ax23 = ax2
ax21r, ax22r, ax23r = ax21.twinx(), ax22.twinx(), ax23.twinx()
ax2r = ax21r, ax22r, ax23r
for ax_ in ax2:
    ax_.spines['left'].set_color('tab:blue')
    ax_.tick_params(axis='y', which='both', colors='tab:blue')
    ax_.yaxis.label.set_color('tab:blue')
    ticks = [1e6 * nn / para.df for nn in range(para.Nbeats + 1)]
    ticks_minor = [1e6 * (nn + 0.5) / para.df for nn in range(para.Nbeats)]
    ax_.set_xticks(ticks)
    ax_.set_xticks(ticks_minor, minor=True)
    for TT in ticks:
        ax_.axvline(TT, ls='--', c='tab:gray')
for ax_ in ax2r:
    ax_.set_ylim(-np.pi, np.pi)
    ax_.spines['right'].set_color('tab:orange')
    ax_.tick_params(axis='y', which='both', colors='tab:orange')
    ax_.yaxis.label.set_color('tab:orange')
    ax_.set_yticks([-np.pi, 0., np.pi])
    ax_.set_yticks([-np.pi / 2, np.pi / 2], minor=True)
    ax_.set_yticklabels([u'\u2212\u03c0', '0', u'\u002b\u03c0'])
ax21.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(drive_envelope), c='tab:blue')
ax21r.plot(1e6 * t_envelope, np.angle(drive_envelope), c='tab:orange', alpha=0.5)
ax22.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(V1_envelope), c='tab:blue')
ax22r.plot(1e6 * t_envelope, np.angle(V1_envelope), c='tab:orange', alpha=0.5)
ax23.plot(1e6 * t_envelope, 1e6 * 2 * np.abs(V2_envelope), c='tab:blue')
ax23r.plot(1e6 * t_envelope, np.angle(V2_envelope), c='tab:orange', alpha=0.5)
ax21.set_ylabel(r"$A_\mathrm{D}$ [$\mathrm{\mu V}$]")
ax22.set_ylabel(r"$A_1$ [$\mathrm{\mu V}$]")
ax23.set_ylabel(r"$A_2$ [$\mathrm{\mu V}$]")
ax21r.set_ylabel(r"$\phi_\mathrm{D}$ [rad]")
ax22r.set_ylabel(r"$\phi_1$ [rad]")
ax23r.set_ylabel(r"$\phi_2$ [rad]")
ax21.set_title(PULSE)
ax23.set_xlabel(r"Time [$\mathrm{\mu s}$]")
fig2.show()
