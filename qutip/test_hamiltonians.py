import time

import numpy as np
import qutip as qt

# HAMILTONIAN = "JC"
# HAMILTONIAN = "RWA"
HAMILTONIAN = "DISP"

amp = 6.35e-1 / np.sqrt(10)
phase = 0.  # OBS: not implemented!
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
N = 15
a = qt.tensor(qt.destroy(N), qt.qeye(2))
nc = a.dag() * a
xc = 0.5 * (a + a.dag())
yc = -0.5j * (a - a.dag())

# qubit operators
sigmam = qt.tensor(qt.qeye(N), qt.sigmam())
sigmap = qt.tensor(qt.qeye(N), qt.sigmap())
sigmax = qt.tensor(qt.qeye(N), qt.sigmax())
sigmay = qt.tensor(qt.qeye(N), qt.sigmay())
sigmaz = qt.tensor(qt.qeye(N), qt.sigmaz())

I = qt.tensor(qt.qeye(N), qt.qeye(2))

# Hamiltonian
wrot1 = 0.  # wc
wrot2 = 0.  # wq

H0 = (wc - wrot1) * (a.dag() * a + I / 2) + 0.5 * (wq - wrot2) * sigmaz

if HAMILTONIAN == "JC":
    V = g * (a.dag() + a) * sigmax
elif HAMILTONIAN == "RWA":
    V = g * (a * sigmap + a.dag() * sigmam)
elif HAMILTONIAN == "DISP":
    V = chi * (a.dag() * a + I / 2) * sigmaz
else:
    raise NotImplementedError

# Hd = 0.5 * amp * (a * np.exp(1j * phase) + a.dag() * np.exp(-1j * phase))
# Hd = 0.5 * amp * (a + a.dag())
# H = H0 + V + Hd

H = [H0, V, [a, 'A * sin(chi * t)**2 * np.exp(1j * wd * t)'], [a.dag(), 'A * sin(chi * t)**2 * np.exp(-1j * wd * t)']]

# Initial state
psi0_g = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 1))  # |g>
psi0_e = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 0))  # |e>
psi0_p = qt.tensor(qt.coherent(N, 0.),
                   (qt.basis(2, 1) + qt.basis(2, 0)).unit())  # |+>

# Time evolution
df = 2. * np.abs(chi) / (2. * np.pi)
# fs = 10 * fc
fs = 2048 * df
dt = 1. / fs
ns = int(round(fs / df))
df = fs / ns
T = 1. / df
Nr = 0
Np = 2
Nt = Nr + Np
tlist = np.linspace(0., T * Nt, ns * Nt, endpoint=False)

# References from master equation
print("Calculating deterministic references")
t0 = time.time()

res = qt.mesolve(
    H,
    psi0_g,
    tlist,
    [np.sqrt(kappa) * a],
    [],
    args={
        'chi': np.abs(chi),
        'A': amp / 2,
        'wd': wc,
    },
)
xref_g = qt.expect(xc, res.states)
yref_g = qt.expect(yc, res.states)

res = qt.mesolve(
    H,
    psi0_e,
    tlist,
    [np.sqrt(kappa) * a],
    [],
    args={
        'chi': np.abs(chi),
        'A': amp / 2,
        'wd': wc,
    },
)
xref_e = qt.expect(xc, res.states)
yref_e = qt.expect(yc, res.states)

ref_g = xref_g + 1j * yref_g
ref_e = xref_e + 1j * yref_e
template = np.conj(ref_e - ref_g)

t1 = time.time()
print(t1 - t0)
