import time

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

amp = 6.35e-1
phase = np.pi / 2
fc = 6.  # resonator frequency
wc = 2. * np.pi * fc

fq = 5.  # qubit frequency
wq = 2. * np.pi * fq

chi = 0.02 * 2. * np.pi
delta = wq - wc
g = np.sqrt(np.abs(chi * delta))
print("delta / g = {}".format(delta / g))

# Q = 100.
# kappa = wc / Q
kappa = chi / 10
Q = wc / kappa
print("kappa = {:.2g} MHz".format(kappa))
print("Q = {:.2g}".format(Q))

# cavity operators
N = 30
a = qt.tensor(qt.destroy(N), qt.qeye(2))
nc = a.dag() * a
xc = 0.5 * (a + a.dag())
yc = -0.5j * (a - a.dag())

# qubit operators
sm = qt.tensor(qt.qeye(N), qt.sigmam())
sz = qt.tensor(qt.qeye(N), qt.sigmaz())
sx = qt.tensor(qt.qeye(N), qt.sigmax())

I = qt.tensor(qt.qeye(N), qt.qeye(2))

# Hamiltonian
wrot = wc
H0 = (wc - wrot) * (a.dag() * a + I / 2) + 0.5 * (wq - wrot) * sz
# V = g * (a * sm.dag() + a.dag() * sm)
V = chi * (a.dag() * a + I / 2) * sz
# Hd = 0.5 * amp * (a * np.exp(1j * phase) + a.dag() * np.exp(-1j * phase))
Hd = 0.5 * amp * (a + a.dag())

# H = H0 + V + Hd
H = [H0, V, [a, 'A * sin(chi * t)**2'], [a.dag(), 'A * sin(chi * t)**2']]

# Initial state
psi0 = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 0))  # excited
# psi0 = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 1))  # ground
# psi0 = qt.tensor(qt.coherent(N, 0.), (qt.basis(2, 1) + qt.basis(2, 0)).unit())

# Time evolution
fs = 10 * fc
dt = 1. / fs
df = 2. * np.abs(chi) / (2. * np.pi)
ns = int(round(fs / df))
df = fs / ns
T = 1. / df
Nr = 0
Np = 2
Nt = Nr + Np
# tlist = np.linspace(0., 250., 1000)
tlist = np.linspace(0., T * Nt, ns * Nt, endpoint=False)
t0 = time.time()
# res = qt.mesolve(H, psi0, tlist, [np.sqrt(kappa) * a], [], options=qt.Odeoptions(nsteps=5000))
res = qt.mesolve(H, psi0, tlist, [np.sqrt(kappa) * a], [], args={'chi': np.abs(chi), 'A': amp / 2}, options=qt.Odeoptions(nsteps=5000))
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

if False:
    # Correlation functions
    tcorr = np.linspace(0, 1000, 10000)
    corr_vec_c = qt.correlation_2op_2t(H, psi0, None, tcorr, [], a.dag(), a)
    corr_vec_q = qt.correlation_2op_2t(H, psi0, None, tcorr, [], sx, sx)

    fig3 = plt.figure(tight_layout=True)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.plot(tcorr, corr_vec_c.real, label='cavity')
    ax3.plot(tcorr, corr_vec_q.real, label='qubit')
    ax3.legend()
    fig3.show()


    # Spectra
    w, S_c = qt.spectrum_correlation_fft(tcorr, corr_vec_c)
    w, S_q = qt.spectrum_correlation_fft(tcorr, corr_vec_q)

    fig4 = plt.figure(tight_layout=True)
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.plot(w / (2 * np.pi), np.abs(S_c), label='cavity')
    ax4.plot(w / (2 * np.pi), np.abs(S_q), label='qubit')
    ax4.legend()
    fig4.show()

# cavity state distribution
rho_c = qt.ptrace(res.states[ns], 0)
xvec = np.linspace(-10, 10, 401)
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

# x_arr = xc_list[-Np * ns:]
# t_arr = tlist[-Np * ns:]
# t_arr -= t_arr[0]
# freqs = np.fft.rfftfreq(Np * ns, dt)
# x_fft = np.fft.rfft(x_arr) / (Np * ns)

# fig4, ax4 = plt.subplots(tight_layout=True)
# ax4.semilogy(freqs, np.abs(x_fft))
# fig4.show()
