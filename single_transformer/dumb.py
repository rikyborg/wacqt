import matplotlib.pyplot as plt
import numpy as np

fc = 1e+9
x = 1e+7
df = 1e+6
fs = 2e+10

dt = 1 / fs
T = 1 / df
ns = int(round(fs / df))

t = np.linspace(0, T, ns, endpoint=False)
freqs = np.fft.rfftfreq(ns, dt)

f1 = fc - x
z1 = np.sinc((freqs - f1) / x)
f2 = fc + x
z2 = np.sinc((freqs - f2) / x)

z = z1 + z2
y = np.fft.irfft(z) * ns

tria = np.zeros_like(t)
idx = np.argmin(np.abs(t - 1 / x))
tria[:idx] = np.linspace(0, 1, idx, endpoint=False)
tria[idx:2 * idx] = np.linspace(1, 0, idx, endpoint=False)
w1 = tria * np.cos(2. * np.pi * f1 * t)
w2 = tria * np.cos(2. * np.pi * f2 * t)
w = w1 - w2
w *= 20

s = np.fft.rfft(w) / ns

fig, ax = plt.subplots(2, 1, tight_layout=True)
ax1, ax2 = ax
# ax1.plot(freqs, np.abs(z1))
# ax1.plot(freqs, np.abs(z2))
ax1.plot(freqs, np.abs(z))
ax1.plot(freqs, np.abs(s))
ax2.plot(t, y)
ax2.plot(t, w)
ax1.set_xlim(fc - 3 * x, fc + 3 * x)
fig.show()
