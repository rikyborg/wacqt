import matplotlib.pyplot as plt
import numpy as np

freqs = np.linspace(0, 1000, 10001)

fig, ax = plt.subplots(2, 1, tight_layout=True)
ax1, ax2 = ax

# phase = 0.
for phase in np.linspace(0, np.pi, 10):
    y1 = np.sinc((freqs - 99) / 1)
    y2 = np.sinc((freqs - 101) / 1) * np.exp(1j * phase)
    y = y1 + y2

    delay = 5000
    x1 = np.fft.irfft(y1 * np.exp(1j * delay * freqs))
    x2 = np.fft.irfft(y2 * np.exp(1j * delay * freqs))
    x = np.fft.irfft(y * np.exp(1j * delay * freqs))

    # ax1.plot(freqs, np.abs(y1))
    # ax1.plot(freqs, np.abs(y2))
    ax1.plot(freqs, np.abs(y))
    # ax2.plot(x1)
    # ax2.plot(x2)
    ax2.plot(x)
fig.show()
