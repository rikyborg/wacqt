import os
import time

import numpy as np
import qutip as qt


def format_sec(s):
    """ Utility function to format a time interval in seconds
    into a more human-readable string.

    Args:
        s (float): time interval in seconds

    Returns:
        (str): time interval in the form "X h Y m Z.z s"

    Examples:
        >>> format_sec(12345.6)
        '3h 25m 45.6s'
    """
    if s < 1.:
        return "{:.1f}ms".format(s * 1e3)

    h = int(s // 3600)
    s -= h * 3600.

    m = int(s // 60)
    s -= m * 60

    if h:
        res = "{:d}h {:d}m {:.1f}s".format(h, m, s)
    elif m:
        res = "{:d}m {:.1f}s".format(m, s)
    else:
        res = "{:.1f}s".format(s)

    return res


class DummyProgressBar():
    def start(*args, **kwargs):
        pass

    def update(*args, **kwargs):
        pass

    def finished(*args, **kwargs):
        pass


class MyProgressBar():
    def start(self, Nsteps):
        self.ii = 0
        self.Nsteps = Nsteps
        self.t_start = time.time()
        self.t_prev = self.t_start

    def update(self):
        self.ii += 1
        t_now = time.time()
        t_run = t_now - self.t_prev
        t_total = t_now - self.t_start
        t_avg = t_total / self.ii
        t_remaining = t_avg * (self.Nsteps - self.ii)
        print("{:d}/{:d} -- Run: {:s} -- Total: {:s} -- Remaining: {:s}{:s}\r".
              format(self.ii, self.Nsteps, format_sec(t_run),
                     format_sec(t_total), format_sec(t_remaining), " " * 20),
              end="")
        self.t_prev = t_now

    def finished(self):
        t_stop = time.time()
        print("Total run time: {:s}{}".format(
            format_sec(t_stop - self.t_start), " " * 100))


USE_SSE = True  # stochastic Schrödinger equation, otherwise stochastic master equation
Ntraj = 65536

amp = 6.35e-1
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
N = 30
a = qt.tensor(qt.destroy(N), qt.qeye(2))
nc = a.dag() * a
xc = 0.5 * (a + a.dag())
yc = -0.5j * (a - a.dag())

# qubit operators
sm = qt.tensor(qt.qeye(N), qt.sigmam())
sz = qt.tensor(qt.qeye(N), qt.sigmaz())
sx = qt.tensor(qt.qeye(N), qt.sigmax())
sy = qt.tensor(qt.qeye(N), qt.sigmay())

I = qt.tensor(qt.qeye(N), qt.qeye(2))

# Hamiltonian
wrot1 = wc
wrot2 = wq
H0 = (wc - wrot1) * (a.dag() * a + I / 2) + 0.5 * (wq - wrot2) * sz
# V = g * (a * sm.dag() + a.dag() * sm)
V = chi * (a.dag() * a + I / 2) * sz
# Hd = 0.5 * amp * (a * np.exp(1j * phase) + a.dag() * np.exp(-1j * phase))
Hd = 0.5 * amp * (a + a.dag())

# H = H0 + V + Hd
H = [H0, V, [a, 'A * sin(chi * t)**2'], [a.dag(), 'A * sin(chi * t)**2']]

# Initial state
psi0_g = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 1))  # |g>
psi0_e = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 0))  # |e>
psi0_p = qt.tensor(qt.coherent(N, 0.),
                   (qt.basis(2, 1) + qt.basis(2, 0)).unit())  # |+>

# Time evolution
df = 2. * np.abs(chi) / (2. * np.pi)
# fs = 10 * fc
fs = 128 * df
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
        'A': amp / 2
    },
)
xref_g = qt.expect(xc, res.states)
yref_g = qt.expect(yc, res.states)

res = qt.mesolve(H,
                 psi0_e,
                 tlist, [np.sqrt(kappa) * a], [],
                 args={
                     'chi': np.abs(chi),
                     'A': amp / 2
                 },
                 options=qt.Odeoptions(nsteps=5000))
xref_e = qt.expect(xc, res.states)
yref_e = qt.expect(yc, res.states)

ref_g = xref_g + 1j * yref_g
ref_e = xref_e + 1j * yref_e
template = np.conj(ref_e - ref_g)

t1 = time.time()
print(format_sec(t1 - t0))
print()

# Fidelity for basis states

print("Fidelity ground state")
scores_g = np.zeros(Ntraj)
avg_traj_g = np.zeros(len(tlist), np.complex128)
correct = 0
bar = MyProgressBar()
bar.start(Ntraj)
for ii in range(Ntraj):
    if USE_SSE:
        result = qt.ssesolve(
            H,
            psi0_g,
            tlist,
            [np.sqrt(kappa) * a],
            [],
            ntraj=1,
            nsubsteps=10,
            solver='taylor15',
            method='heterodyne',
            store_measurement=True,
            args={
                'chi': np.abs(chi),
                'A': amp / 2
            },
            progress_bar=DummyProgressBar(),
        )
    else:
        result = qt.smesolve(
            H,
            psi0_g,
            tlist,
            [],
            [np.sqrt(kappa) * a],
            [],
            ntraj=1,
            nsubsteps=10,
            solver='taylor15',
            method='heterodyne',
            store_measurement=True,
            args={
                'chi': np.abs(chi),
                'A': amp / 2
            },
            progress_bar=DummyProgressBar(),
        )
    measurement = result.measurement[0]
    m = measurement[:, 0, 0].real + 1j * measurement[:, 0, 1].real
    avg_traj_g += m
    score = np.sum(template * m).real
    if score < 0:
        correct += 1
    else:
        # print("WRONG!!! ", end="")
        pass
    scores_g[ii] = score

    bar.update()

bar.finished()

avg_traj_g /= Ntraj
print("{:d} out of {:d}: {:.1%}".format(correct, Ntraj, correct / Ntraj))
print()

print("Fidelity excited state")
scores_e = np.zeros(Ntraj)
avg_traj_e = np.zeros(len(tlist), np.complex128)
correct = 0
bar = MyProgressBar()
bar.start(Ntraj)
for ii in range(Ntraj):
    if USE_SSE:
        result = qt.ssesolve(
            H,
            psi0_e,
            tlist,
            [np.sqrt(kappa) * a],
            [],
            ntraj=1,
            nsubsteps=10,
            solver='taylor15',
            method='heterodyne',
            store_measurement=True,
            args={
                'chi': np.abs(chi),
                'A': amp / 2
            },
            progress_bar=DummyProgressBar(),
        )
    else:
        result = qt.smesolve(
            H,
            psi0_e,
            tlist,
            [],
            [np.sqrt(kappa) * a],
            [],
            ntraj=1,
            nsubsteps=10,
            solver='taylor15',
            method='heterodyne',
            store_measurement=True,
            args={
                'chi': np.abs(chi),
                'A': amp / 2
            },
            progress_bar=DummyProgressBar(),
        )
    measurement = result.measurement[0]
    m = measurement[:, 0, 0].real + 1j * measurement[:, 0, 1].real
    avg_traj_e += m
    score = np.sum(template * m).real
    if score > 0:
        correct += 1
    else:
        print("WRONG!!! ", end="")
    scores_e[ii] = score

    bar.update()

bar.finished()

avg_traj_e /= Ntraj
t1 = time.time()
print("{:d} out of {:d}: {:.1%}".format(correct, Ntraj, correct / Ntraj))
print()

print("Fidelity plus state")
ssz = np.zeros((Ntraj, Np * ns))
ssx = np.zeros((Ntraj, Np * ns))
ssy = np.zeros((Ntraj, Np * ns))
scores_p = np.zeros(Ntraj)
correct = 0
bar = MyProgressBar()
bar.start(Ntraj)
for ii in range(Ntraj):
    if USE_SSE:
        result = qt.ssesolve(
            H,
            psi0_p,
            tlist,
            [np.sqrt(kappa) * a],
            [],
            ntraj=1,
            nsubsteps=10,
            solver='taylor15',
            method='heterodyne',
            store_measurement=True,
            args={
                'chi': np.abs(chi),
                'A': amp / 2
            },
            progress_bar=DummyProgressBar(),
        )
    else:
        result = qt.smesolve(
            H,
            psi0_p,
            tlist,
            [],
            [np.sqrt(kappa) * a],
            [],
            ntraj=1,
            nsubsteps=10,
            solver='taylor15',
            method='heterodyne',
            store_measurement=True,
            args={
                'chi': np.abs(chi),
                'A': amp / 2
            },
            progress_bar=DummyProgressBar(),
        )
    measurement = result.measurement[0]
    m = measurement[:, 0, 0].real + 1j * measurement[:, 0, 1].real
    score = np.sum(template * m).real
    ssz[ii] = qt.expect(sz, result.states[0]).real
    ssx[ii] = qt.expect(sx, result.states[0]).real
    ssy[ii] = qt.expect(sy, result.states[0]).real
    if ssz[ii, -1] > 0:
        if score > 0:
            correct += 1
        else:
            print("WRONG!!! ", end="")
    if ssz[ii, -1] < 0:
        if score < 0:
            correct += 1
        else:
            print("WRONG!!! ", end="")
    scores_p[ii] = score

    bar.update()

bar.finished()

print("{:d} out of {:d}: {:.1%}".format(correct, Ntraj, correct / Ntraj))
print()

scriptname = os.path.splitext(os.path.basename(__file__))[0]
save_filename = "{:s}_{:d}_{:s}_g_e_p_{:d}.npz".format(
    scriptname, int(time.time()), "SSE" if USE_SSE else "SME", Ntraj)
np.savez(
    save_filename,
    amp=amp,
    phase=0.,
    wc=wc,
    wq=wq,
    chi=chi,
    kappa=kappa,
    N=N,
    wrot1=wrot1,
    wrot2=wrot2,
    ns=ns,
    Np=Np,
    Ntraj=Ntraj,
    scores_g=scores_g,
    scores_e=scores_e,
    scores_p=scores_p,
    ssz=ssz,
    ssx=ssx,
    ssy=ssy,
    tlist=tlist,
    ref_g=ref_g,
    ref_e=ref_e,
    avg_traj_g=avg_traj_g,
    avg_traj_e=avg_traj_e,
)
