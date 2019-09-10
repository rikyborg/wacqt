import matplotlib.pyplot as plt
import numpy as np

fc = 1e+9
fm = 1e+7
fm2 = 1e+8
df = 1e+6
fs = 2e+10

dt = 1 / fs
T = 1 / df
ns = int(round(fs / df))

wc = 2 * np.pi * fc
wm = 2 * np.pi * fm
wm2 = 2 * np.pi * fm2

t = np.linspace(0, T, ns, endpoint=False)
freqs = np.fft.rfftfreq(ns, dt)

carrier = np.cos(wc * t)
mask1 = np.zeros_like(t)
T0, T1 = 0., 0.5 / fm
mask1[np.logical_and(t >= T0, t < T1)] = 1.
s1 = np.sin(wm * t)**3 * carrier
s1 *= mask1
s1_fft = np.fft.rfft(s1) / ns

T2 = T1 + 0.5 / fm2
mask2 = np.zeros_like(t)
mask2[np.logical_and(t >= T1, t < T2)] = 1.
s2 = np.sin(wm2 * t)**3 * carrier
s2 *= mask2
s2_fft = np.fft.rfft(s2) / ns

idx = int(round(fc / df))
A = - s1_fft[idx] / s2_fft[idx]
s0_fft = s1_fft + A * s2_fft


fig1, ax1 = plt.subplots(2, 1, tight_layout=True)
ax11, ax12 = ax1

ax11.plot(t, s1)
ax11.plot(t, s2)
ax12.plot(freqs, np.abs(s1_fft))
ax12.plot(freqs, np.abs(s2_fft))
ax12.plot(freqs, np.abs(s0_fft))

fig1.show()
