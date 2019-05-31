import time

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

amp = 1e-1
phase = np.pi / 2
fc = 6.  # resonator frequency
wc = 2. * np.pi * fc
Q = 100.
kappa = wc / Q

fq = 5.  # qubit frequency
wq = 2. * np.pi * fq

chi = kappa
delta = wq - wc
g = np.sqrt(np.abs(chi * delta))

# cavity operators
N = 30
a = qt.tensor(qt.destroy(N), qt.qeye(2))
nc = a.dag() * a
xc = 0.5 * (a + a.dag())
yc = -0.5j * (a - a.dag())

# qubit operators
sm = qt.tensor(qt.qeye(N), qt.sigmam())
sz = qt.tensor(qt.qeye(N), qt.sigmaz())

I = qt.tensor(qt.qeye(N), qt.qeye(2))

# Hamiltonian
wrot = wc
H0 = (wc - wrot) * (a.dag() * a) + 0.5 * (wq - wrot) * sz
# V = g * (a * sm.dag() + a.dag() * sm)
V = chi * (a.dag() * a + I / 2) * sz
# Hd = 0.5 * amp * (a * np.exp(1j * phase) + a.dag() * np.exp(-1j * phase))
Hd = 0.5 * amp * (a + a.dag())

H = H0 + V
# H = [H0, [a, 'A * exp(1j*wd*t)'], [a.dag(), 'A * exp(-1j*wd*t)']]

# Initial state
psi0 = qt.tensor(qt.coherent(N, 2.), qt.basis(2, 1))

# Time evolution
fs = 10 * fc
dt = 1. / fs
df = kappa / (2. * np.pi)
ns = int(round(fs / df))
df = fs / ns
T = 1. / df
Nr = 0
Np = 10
Nt = Nr + Np
# tlist = np.linspace(0., 500., 5000)
tlist = np.linspace(0., T * Nt, ns * Nt, endpoint=False)
t0 = time.time()
res = qt.mesolve(H, psi0, tlist, [np.sqrt(kappa) * a], [], args={'wd': wc, 'A': amp / 2})
t1 = time.time()
print(t1 - t0)

# Excitation numbers
nc_list = qt.expect(nc, res.states)
sz_list = qt.expect(sz, res.states)

fig1 = plt.figure(tight_layout=True)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(tlist, nc_list, label='cavity')
ax1.plot(tlist, sz_list, label='qubit')
ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("<n>")
ax1.legend()
fig1.show()

# Resonator quadratures
xc_list = qt.expect(xc, res.states)
yc_list = qt.expect(yc, res.states)

fig2 = plt.figure(tight_layout=True)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(tlist, xc_list, label='cavity x')
ax2.plot(tlist, yc_list, label='cavity y')
ax2.set_xlabel("Time [ns]")
ax2.set_ylabel("<x>")
ax2.legend()
fig2.show()

# cavity state distribution
rho_c = qt.ptrace(res.states[-1], 0)
xvec = np.linspace(-5, 5, 200)
W_c = qt.wigner(rho_c, xvec, xvec)
wlim = np.abs(W_c).max()

fig5 = plt.figure()
ax5 = fig5.add_subplot(1, 1, 1)
# ax5.contourf(xvec, xvec, W_c, 100, cmap='RdBu', vmin=-wlim, vmax=wlim)
ax5.imshow(
    W_c,
    cmap='RdBu_r',
    vmin=-wlim,
    vmax=wlim,
    origin='bottom',
    extent=(xvec[0], xvec[-1], xvec[0], xvec[-1]))
ax5.axhline(0, ls='--', c='tab:gray', alpha=0.5)
ax5.axvline(0, ls='--', c='tab:gray', alpha=0.5)
ax5.set_aspect('equal')
ax5.set_xlabel(r'Re$\alpha$')
ax5.set_ylabel(r'Im$\alpha$')
fig5.show()

x_arr = xc_list[-Np * ns:]
t_arr = tlist[-Np * ns:]
t_arr -= t_arr[0]
freqs = np.fft.rfftfreq(Np * ns, dt)
x_fft = np.fft.rfft(x_arr) / (Np * ns)

fig4, ax4 = plt.subplots(tight_layout=True)
ax4.semilogy(freqs, np.abs(x_fft))
fig4.show()
