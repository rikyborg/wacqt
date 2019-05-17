import time

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

amp = 1e-1
# phase = np.pi / 2
wr = 2. * np.pi * 1  # resonator frequency
kappa = 0.06

# cavity operators
N = 30
a = qt.destroy(N)
nc = a.dag() * a
xc = 0.5 * (a + a.dag())
yc = -0.5j * (a - a.dag())

I = qt.qeye(N)

# Hamiltonian
H0 = wr * (a.dag() * a)
# H0 = 0. * I
# Hd = 0.5 * amp * (a * np.exp(1j * phase) + a.dag() * np.exp(-1j * phase))
Hd = 0.5 * amp * (a + a.dag())

# H = H0 + Hd
H = [H0, [a, 'A * exp(1j*wd*t)'], [a.dag(), 'A * exp(-1j*wd*t)']]
# H = [H0, [a + a.dag(), 'A * sin(wd*t)']]
# H = [H0, [Hd, '1. - exp(-t / 5)']]
# H = [H0, [Hd, 'exp(-(t / 5) ** 2)']]

# Initial state
psi0 = qt.coherent(N, 0)

# Time evolution
tlist = np.linspace(0., 500., 5000)
t0 = time.time()
res = qt.mesolve(H, psi0, tlist, [np.sqrt(kappa) * a], [], args={'wd': wr, 'A': amp / 2})
t1 = time.time()
print(t1 - t0)

# Excitation numbers
nc_list = qt.expect(nc, res.states)

fig1 = plt.figure(tight_layout=True)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(tlist, nc_list, label='cavity')
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
rho_c = res.states[-1]
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
