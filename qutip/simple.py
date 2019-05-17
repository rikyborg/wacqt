# based on
# https://nbviewer.jupyter.org/github/jrjohansson/qutip-lectures/blob/master/Lecture-10-cQED-dispersive-regime.ipynb
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

N = 20

wr = 2. * np.pi * 2.0  # resonator frequency
wq = 2. * np.pi * 3.0  # qubit frequency
chi = 2. * np.pi * 0.025  # dispersive shift

delta = np.abs(wr - wq)
g = np.sqrt(delta * chi)
print("delta / g = {:.2g}".format(delta / g))

# cavity operators
a = qt.tensor(qt.destroy(N), qt.qeye(2))
nc = a.dag() * a
xc = a + a.dag()

# qubit operators
sm = qt.tensor(qt.qeye(N), qt.sigmam())  # !!! instead of qt.destroy(2)
sz = qt.tensor(qt.qeye(N), qt.sigmaz())
sx = qt.tensor(qt.qeye(N), qt.sigmax())
nq = sm.dag() * sm
xq = sm + sm.dag()

I = qt.tensor(qt.qeye(N), qt.qeye(2))

# Dispersive hamiltonian (hbar = 1)
H = wr * (a.dag() * a + I / 2) + 0.5 * wq * sz + chi * (a.dag() * a + I / 2) * sz

# Initial state
psi0 = qt.tensor(qt.coherent(N, np.sqrt(4)), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
# psi0 = qt.tensor(qt.coherent(N, np.sqrt(4)), qt.basis(2, 1))

# Time evolution
tlist = np.linspace(0, 250, 1000)
res = qt.mesolve(H, psi0, tlist, [], [], options=qt.Odeoptions(nsteps=5000))

# Excitation numbers
nc_list = qt.expect(nc, res.states)
nq_list = qt.expect(nq, res.states)

fig1 = plt.figure(tight_layout=True)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(tlist, nc_list, label='cavity')
ax1.plot(tlist, nq_list, label='qubit')
ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("<n>")
ax1.legend()
fig1.show()

# Resonator quadrature
xc_list = qt.expect(xc, res.states)

fig2 = plt.figure(tight_layout=True)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(tlist, xc_list, label='cavity')
ax2.set_xlabel("Time [ns]")
ax2.set_ylabel("<x>")
ax2.legend()
fig2.show()

# Correlation function for the resonator
# (could also just FFT the previous result, but we would need to simulate longer)
tlist_corr = np.linspace(0, 1000, 10000)
corr_vec_c = qt.correlation(H, psi0, None, tlist_corr, [], a.dag(), a)

fig3 = plt.figure(tight_layout=True)
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(tlist_corr, corr_vec_c.real, label='resonator')
ax3.set_xlabel("Time [ns]")
ax3.set_ylabel("correlation")
ax3.legend()
fig3.show()

w, S_c = qt.spectrum_correlation_fft(tlist_corr, corr_vec_c)

fig4 = plt.figure(tight_layout=True)
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(w / (2. * np.pi), np.abs(S_c), label='cavity')
# ax4.set_xlim((wr - 2.5 * chi) / (2. * np.pi), (wr + 2.5 * chi) / (2. * np.pi))
ax4.set_xlabel('Frequency [GHz]')
ax4.legend()
fig4.show()

# correlation function for the qubit
corr_vec_q = qt.correlation(H, psi0, None, tlist_corr, [], sm.dag(), sm)  # !!! instead of sx, sx
ax3.plot(tlist_corr, corr_vec_q.real, label='qubit')
ax3.legend()
fig3.canvas.draw()

# spectrum of the qubit
w, S_q = qt.spectrum_correlation_fft(tlist_corr, corr_vec_q)
ax4.plot(w / (2. * np.pi), np.abs(S_q), label='qubit')
fig4.canvas.draw()

# cavity state distribution
rho_c = qt.ptrace(res.states[-1], 0)
xvec = np.linspace(-5, 5, 200)
W_c = qt.wigner(rho_c, xvec, xvec)
wlim = np.abs(W_c).max()

fig5 = plt.figure()
ax5 = fig5.add_subplot(1, 1, 1)
ax5.contourf(xvec, xvec, W_c, 100, cmap='RdBu', vmin=-wlim, vmax=wlim)
ax5.set_aspect('equal')
ax5.set_xlabel(r'Re$\alpha$')
ax5.set_ylabel(r'Im$\alpha$')
fig5.show()
