import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


def triang(N, endpoint=False):
    if endpoint:
        N = N - 1
    odd = N % 2
    if odd:
        N = N * 2
    n = N // 2
    up = np.linspace(0.0, 1.0, n, endpoint=False)
    down = np.linspace(1.0, 0.0, n, endpoint=False)
    triang = np.r_[up, down]
    if odd:
        triang = triang[::2]
    if endpoint:
        triang = np.r_[triang, 0.0]
    return triang


# METHOD = "square"
METHOD = "triangle"
# METHOD = "sin2"

ROTFRM = True
# ROTFRM = False

N = 40
wr = 2 * np.pi * 6.0  # GHz, resonator frequency
wq = 2 * np.pi * 5.0  # GHz, qubit frequency
chi = -2 * np.pi * 0.01  # GHz, dispersive shift
kappa = 2 * np.pi * 0.01
fs = 20.0

delta = wq - wr  # resonator-qubit detuning
g = np.sqrt(delta * chi)  # coupling strength

_df = 2 * np.abs(chi) / (2 * np.pi)
ns = int(round(fs / _df))
df = fs / ns
T = 1 / df
dt = 1 / fs

tlist = np.linspace(0, 10 * T, 10 * ns, endpoint=False)


# cavity operators
a = qt.tensor(qt.destroy(N), qt.qeye(2))
nc = a.dag() * a
xc = 0.5 * (a + a.dag())
yc = -0.5j * (a - a.dag())

# atomic operators
sigmam = qt.tensor(qt.qeye(N), qt.sigmam())
sigmap = qt.tensor(qt.qeye(N), qt.sigmap())
sigmax = qt.tensor(qt.qeye(N), qt.sigmax())
sigmay = qt.tensor(qt.qeye(N), qt.sigmay())
sigmaz = qt.tensor(qt.qeye(N), qt.sigmaz())
nq = sigmam.dag() * sigmam

eye = qt.tensor(qt.qeye(N), qt.qeye(2))

# dispersive hamiltonian
if ROTFRM:
    wrot1 = wr
    wrot2 = wq
else:
    wrot1 = 0.0
    wrot2 = 0.0

len_r = 10
wait_r = int(round(2 * np.pi / np.abs(chi) / dt)) // 4
amp_r = 2 * 4.457591364429735
# amp_r = 0.0
wd_r = (wr - wrot1) - chi
pd_r = -np.pi / 2
_drive_r = 0.5 * amp_r * np.exp(+1j * (wd_r * tlist + pd_r))
mask_r = np.zeros_like(tlist)
# First displacement
mask_r[:len_r] = 1.0
# 2nd displacement after C_PI
mask_r[len_r + wait_r:len_r + wait_r + len_r:] = 1.0
start_q = len_r + wait_r + len_r

amp_q = np.pi / T
# amp_q = 0.0
wd_q = (wq - wrot2) + chi + 0 * 2 * chi
pd_q = -np.pi / 2
_drive_q = 0.5 * amp_q * np.exp(+1j * (wd_q * tlist + pd_q))
mask_q = np.zeros_like(tlist)
if METHOD == "triangle":
    len_q = 2 * ns
    mask_q[start_q:start_q + len_q] = triang(2 * ns)
elif METHOD == "square":
    len_q = ns
    mask_q[start_q:start_q + len_q] = 1.0
elif METHOD == "sin2":
    len_q = 2 * ns
    mask_q[start_q:start_q + len_q] = np.sin(0.5 * chi * tlist[:len_q])**2
else:
    raise NotImplementedError

# 3rd displacement
mask_r[start_q + len_q:start_q + len_q + len_r] = -1.0

drive_r = _drive_r * mask_r
drive_q = _drive_q * mask_q

drive_q_fft = np.fft.fft(drive_q + drive_q.conj()) / len(drive_q)
fftfreq = np.fft.fftfreq(len(drive_q), dt)
fftw = 2 * np.pi * fftfreq

H0 = (wr - wrot1) * (a.dag() * a + eye / 2) \
    + 0.5 * (wq - wrot2) * sigmaz
V = chi * (a.dag() * a + eye / 2) * sigmaz
H = [
    H0,
    V,

    [a, drive_r.copy()],
    [a.dag(), drive_r.conj()],

    # [sigmam, 'A * exp(+1j * wd * t)'],
    # [sigmam.dag(), 'A * exp(-1j * wd * t)'],
    [sigmam, drive_q.copy()],
    [sigmam.dag(), drive_q.conj()],
]

psi_0g = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 1))  # |g>
psi_0e = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 0))  # |e>
psi_0p = qt.tensor(qt.coherent(N, 0.),
                   (qt.basis(2, 1) + qt.basis(2, 0)).unit())  # |+>

# psi0 = qt.tensor(
#     qt.coherent(N, np.sqrt(4)),
#     (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
# )
psi0 = qt.tensor(
    # qt.coherent(N, np.sqrt(4)),
    # (qt.basis(N, 0) + qt.basis(N, 1)).unit(),
    qt.basis(N, 0),

    # qt.basis(2, 1),  # |g>
    # qt.basis(2, 0),  # |e>
    (qt.basis(2, 1) + qt.basis(2, 0)).unit(),  # |+>
)
# D2 = qt.tensor(qt.displace(N, 2.0), qt.qeye(2))
# psi0 = D2 * psi0

# time evolution
# tlist = np.linspace(0, 250, 1000)
# res = qt.mesolve(H, psi0, tlist, [np.sqrt(kappa) * a], [], args={'A': amp / 2, 'wd': (wr - wrot1) - chi})
res = qt.mesolve(H, psi0, tlist, [], [])
nc_list = qt.expect(nc, res.states)
nq_list = qt.expect(nq, res.states)

fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True)
ax.plot(tlist / dt, nc_list, label="cavity")
ax.plot(tlist / dt, nq_list, label="qubit")
ax.set_ylabel("n")
ax.set_xlabel("Time (ns)")
ax.legend()
fig.show()

if False:
    # Resonator quadrature
    xc_list = qt.expect(xc, res.states)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4))

    ax.plot(tlist, xc_list, 'r', linewidth=2, label="cavity")
    ax.set_ylabel("x", fontsize=16)
    ax.set_xlabel("Time (ns)", fontsize=16)
    ax.legend()

    fig.tight_layout()
    fig.show()

    # Correlation function for the resonator
    tcorr = np.linspace(0, 10 * T, 10 * ns, endpoint=False)
    corr_vec = qt.correlation_2op_2t(H, psi0, None, tcorr, [], a.dag(), a)

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4))

    ax.plot(tcorr, np.real(corr_vec), 'r', linewidth=2, label="resonator")
    ax.set_ylabel("correlation", fontsize=16)
    ax.set_xlabel("Time (ns)", fontsize=16)
    ax.legend()
    ax.set_xlim(0, 50)
    fig.tight_layout()
    fig.show()

    # Spectrum of the resonator
    w, S = qt.spectrum_correlation_fft(tcorr, corr_vec)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(w / (2 * np.pi), np.abs(S))
    ax.set_xlabel(r'$\omega$', fontsize=18)
    ax.set_xlim(wr/(2*np.pi)-.5, wr/(2*np.pi)+.5)
    fig.show()

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot((w-wr)/chi, np.abs(S))
    ax.set_xlabel(r'$(\omega-\omega_r)/\chi$', fontsize=18)
    ax.set_xlim(-2, 2)
    fig.show()

    # Correlation function of the qubit
    corr_vec = qt.correlation_2op_2t(H, psi0, None, tcorr, [], sigmax, sigmax)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(tcorr, np.real(corr_vec), label="qubit")
    ax.set_ylabel("correlation")
    ax.set_xlabel("Time (ns)")
    ax.legend()
    fig.show()

    # Spectrum of the qubit
    w, S = qt.spectrum_correlation_fft(tcorr, corr_vec)
    S /= len(corr_vec)

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(w / (2 * np.pi), np.abs(S))
    ax.plot(fftfreq, np.abs(drive_q_fft))
    ax.set_xlabel(r'Frequency [GHz]')
    fig.show()

    fig, ax = plt.subplots(tight_layout=True)
    ax.plot((w - wq - chi) / np.abs(2 * chi), np.abs(S))
    ax.plot((fftw - wq - chi) / np.abs(2 * chi), np.abs(drive_q_fft))
    ax.set_xlabel(r'$(\omega - \omega_q - \chi)/2\chi$')
    ax.set_xlim(-N - 0.5, 0.5)
    fig.show()

# Cavity Wigner function
for idx in [0,
            len_r,
            len_r + wait_r,
            len_r + wait_r + len_r,
            len_r + wait_r + len_r + len_q,
            len_r + wait_r + len_r + len_q + len_r,
]:
    # rho_c = qt.ptrace(D2 * res.states[idx], 0)
    rho_c = qt.ptrace(res.states[idx], 0)
    xvec = np.linspace(-10, 10, 501)
    W_c = qt.wigner(rho_c, xvec, xvec)
    wlim = np.abs(W_c).max()

    fig, ax = plt.subplots(tight_layout=True)
    # ax5.contourf(xvec, xvec, W_c, 100, cmap='RdBu', vmin=-wlim, vmax=wlim)
    ax.imshow(
        W_c,
        cmap='RdBu_r',
        vmin=-wlim,
        vmax=wlim,
        origin='bottom',
        extent=(xvec[0], xvec[-1], xvec[0], xvec[-1]))
    ax.axhline(0, ls='--', c='tab:gray', alpha=0.5)
    ax.axvline(0, ls='--', c='tab:gray', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Re$\alpha$')
    ax.set_ylabel(r'Im$\alpha$')
    ax.set_title(f"{idx:d}")
    fig.show()
