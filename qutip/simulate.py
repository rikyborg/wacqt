import os
import sys
import time

import numpy as np
import qutip as qt

if len(sys.argv) == 1:
    # PULSE = "short single"
    # PULSE = "double"
    # PULSE = "cool"
    # PULSE = "single"
    # PULSE = "almost"  # OBS: double length (4 pulses)!
    # PULSE = "long double"
    PULSE = "long single"

    chi_over_kappa = 10.

elif len(sys.argv) == 3:
    PULSE = str(sys.argv[1])
    chi_over_kappa = float(sys.argv[2])

else:
    print("Either no arguments, or PULSE and chi_over_kappa!")
    sys.exit(1)

USE_SSE = True  # stochastic Schrödinger eq., otherwise stochastic master eq.

# HAMILTONIAN = "JC"
# HAMILTONIAN = "RWA"
HAMILTONIAN = "DISP"

ENDPOINT = True

Ntraj = 32768

MAX_PH = 10.0
_amp = 1e-1  # initial amplitude
phase = 0.  # OBS: not implemented!

fc = 6.  # resonator frequency
fq = 5.2  # qubit frequency
chi = 0.002 * 2. * np.pi  # dispersive shift


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


def inprod(f, g, t=None, dt=None):
    if t is not None:
        dt = t[1] - t[0]
        ns = len(t)
        T = ns * dt
    elif dt is not None:
        ns = len(f)
        T = ns * dt
    else:
        T = 1.
    return np.trapz(f * np.conj(g), x=t) / T


def norm(x, t=None, dt=None):
    return np.sqrt(np.real(inprod(x, x, t=t, dt=dt)))


def proj(v, u, t=None, dt=None):
    if not norm(u, t=t, dt=dt):
        return 0.
    else:
        return inprod(v, u, t=t, dt=dt) / inprod(u, u, t=t, dt=dt) * u


def dist(f, g, t=None, dt=None):
    return norm(f - g, t=t, dt=dt)


wc = 2. * np.pi * fc
wq = 2. * np.pi * fq
delta = wq - wc
g = np.sqrt(np.abs(chi * delta))
ncrit = delta**2 / g**2 / 4
print("delta / g = {}".format(delta / g))
print("ncrit = {}".format(ncrit))
assert MAX_PH < ncrit

# Q = 100.
# kappa = wc / Q
# kappa = chi / 10
kappa = chi / chi_over_kappa
Q = wc / kappa
print("kappa = {:.2g} GHz".format(kappa / 2 / np.pi))
print("Q = {:.2g}".format(Q))

# cavity operators
N = 30
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
wrot1 = wc
wrot2 = wq

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

# Initial state
psi0_g = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 1))  # |g>
psi0_e = qt.tensor(qt.coherent(N, 0.), qt.basis(2, 0))  # |e>
psi0_p = qt.tensor(qt.coherent(N, 0.),
                   (qt.basis(2, 1) + qt.basis(2, 0)).unit())  # |+>

# Time evolution
T = 2 * np.pi / chi / 2  # single pulse
df = 1 / T
# fs = 10 * fc
fs = 128 * df
dt = 1. / fs
ns = int(round(fs / df))
# df = fs / ns
# T = 1. / df
if PULSE in ["almost", "long double", "long single"]:
    Np = 4
else:
    Np = 2  # number of pulses
Ns = ns * Np  # number of time samples
if ENDPOINT:
    Ns += 1
tlist = np.linspace(0., T * Np, Ns, endpoint=ENDPOINT)


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


# References from master equation
print("Calculating deterministic references")
t0 = time.time()

for ii in range(5):
    amp = _amp
    if PULSE == "cool":
        drive = 0.5 * amp * np.sin(3 * chi * tlist) * triang(Ns, ENDPOINT)
    elif PULSE == "almost":
        drive = 0.5 * amp * np.sin(chi * tlist)**2 * triang(Ns, ENDPOINT)
    elif PULSE == "double":
        drive = 0.5 * amp * np.sin(chi * tlist)**2
    elif PULSE == "long double":
        drive = 0.5 * amp * np.sin(0.5 * chi * tlist)**2
    elif PULSE == "short single":
        drive = np.zeros_like(tlist)
        drive[:Ns // 2] = 0.5 * amp * np.sin(chi * tlist[:Ns // 2])**2
    elif PULSE == "single":
        drive = 0.5 * amp * np.sin(0.5 * chi * tlist)**2
    elif PULSE == "long single":
        drive = 0.5 * amp * np.sin(0.25 * chi * tlist)**2
    else:
        raise NotImplementedError

    H = [
        H0,
        V,
        [a, drive.copy()],
        [a.dag(), drive.copy()],
    ]
    res_g = qt.mesolve(
        H,
        psi0_g,
        tlist,
        [np.sqrt(kappa) * a],
        [],
    )
    nmax_g = np.max(qt.expect(nc, res_g.states))

    res_e = qt.mesolve(
        H,
        psi0_e,
        tlist,
        [np.sqrt(kappa) * a],
        [],
    )
    nmax_e = np.max(qt.expect(nc, res_e.states))

    nmax = max(nmax_g, nmax_e)
    _amp = amp * np.sqrt(MAX_PH / nmax)
    print(nmax)

xref_g = qt.expect(xc, res_g.states)
yref_g = qt.expect(yc, res_g.states)
nph_g = qt.expect(nc, res_g.states)
xref_e = qt.expect(xc, res_e.states)
yref_e = qt.expect(yc, res_e.states)
nph_e = qt.expect(nc, res_e.states)
print("Leftover: {:.1e}".format(max(nph_g[-1], nph_e[-1])))

template_g = np.sqrt(kappa) * (xref_g + 1j * yref_g)
template_e = np.sqrt(kappa) * (xref_e + 1j * yref_e)
template_diff = template_e - template_g
threshold = 0.5 * (norm(template_e, tlist)**2 - norm(template_g, tlist)**2)

t1 = time.time()
print(format_sec(t1 - t0))
print()

# Fidelity for basis states

print("Fidelity ground state")
scores_g = np.zeros(Ntraj)
avg_traj_g = np.zeros(len(tlist),
                      np.complex256)  # use extended-precision accumulator
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
            progress_bar=DummyProgressBar(),
        )
    measurement = result.measurement[0]
    m = 0.5 * (measurement[:, 0, 0].real + 1j * measurement[:, 0, 1].real)
    # divide by two because they use x = a + a.dag(),
    # but we want x = (a + a.dag()) / 2
    avg_traj_g += m
    score = np.real(inprod(m, template_diff, tlist))
    if score < threshold:
        correct += 1
    else:
        # print("WRONG!!! ", end="")
        pass
    scores_g[ii] = score

    bar.update()

bar.finished()

avg_traj_g /= Ntraj
avg_traj_g = avg_traj_g.astype(np.complex128)  # cast back to double precision
print("{:d} out of {:d}: {:.1%}".format(correct, Ntraj, correct / Ntraj))
print()

print("Fidelity excited state")
scores_e = np.zeros(Ntraj)
avg_traj_e = np.zeros(len(tlist),
                      np.complex256)  # use extended-precision accumulator
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
            progress_bar=DummyProgressBar(),
        )
    measurement = result.measurement[0]
    m = 0.5 * (measurement[:, 0, 0].real + 1j * measurement[:, 0, 1].real)
    # divide by two because they use x = a + a.dag(),
    # but we want x = (a + a.dag()) / 2
    avg_traj_e += m
    score = np.real(inprod(m, template_diff, tlist))
    if score > threshold:
        correct += 1
    else:
        print("WRONG!!! ", end="")
    scores_e[ii] = score

    bar.update()

bar.finished()

avg_traj_e /= Ntraj
avg_traj_e = avg_traj_e.astype(np.complex128)  # cast back to double precision
t1 = time.time()
print("{:d} out of {:d}: {:.1%}".format(correct, Ntraj, correct / Ntraj))
print()

# print("Fidelity plus state")
# ssz = np.zeros((Ntraj, Np * ns))
# ssx = np.zeros((Ntraj, Np * ns))
# ssy = np.zeros((Ntraj, Np * ns))
# scores_p = np.zeros(Ntraj)
# correct = 0
# bar = MyProgressBar()
# bar.start(Ntraj)
# for ii in range(Ntraj):
#     if USE_SSE:
#         result = qt.ssesolve(
#             H,
#             psi0_p,
#             tlist,
#             [np.sqrt(kappa) * a],
#             [],
#             ntraj=1,
#             nsubsteps=10,
#             solver='taylor15',
#             method='heterodyne',
#             store_measurement=True,
#             # args={
#             #     'chi': np.abs(chi),
#             #     'A': amp / 2
#             # },
#             progress_bar=DummyProgressBar(),
#         )
#     else:
#         result = qt.smesolve(
#             H,
#             psi0_p,
#             tlist,
#             [],
#             [np.sqrt(kappa) * a],
#             [],
#             ntraj=1,
#             nsubsteps=10,
#             solver='taylor15',
#             method='heterodyne',
#             store_measurement=True,
#             # args={
#             #     'chi': np.abs(chi),
#             #     'A': amp / 2
#             # },
#             progress_bar=DummyProgressBar(),
#         )
#     measurement = result.measurement[0]
#     m = 0.5 * (measurement[:, 0, 0].real + 1j * measurement[:, 0, 1].real)
#     # divide by two because they use x = a + a.dag(),
#     # but we want x = (a + a.dag()) / 2
#     score = np.real(inprod(m, template_diff, tlist))
#     ssz[ii] = qt.expect(sigmaz, result.states[0]).real
#     ssx[ii] = qt.expect(sigmax, result.states[0]).real
#     ssy[ii] = qt.expect(sigmay, result.states[0]).real
#     if ssz[ii, -1] > 0:
#         if score > threshold:
#             correct += 1
#         else:
#             print("WRONG!!! ", end="")
#     if ssz[ii, -1] < 0:
#         if score < threshold:
#             correct += 1
#         else:
#             print("WRONG!!! ", end="")
#     scores_p[ii] = score

#     bar.update()

# bar.finished()

# print("{:d} out of {:d}: {:.1%}".format(correct, Ntraj, correct / Ntraj))
# print()

scriptname = os.path.splitext(os.path.basename(__file__))[0]
struct_time = time.localtime()
save_filename = "{:s}_{:s}_{:g}_{:d}.npz".format(
    scriptname,
    time.strftime("%Y%m%d_%H%M%S", struct_time),
    chi_over_kappa,
    Ntraj,
)
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
    # scores_p=scores_p,
    # ssz=ssz,
    # ssx=ssx,
    # ssy=ssy,
    tlist=tlist,
    drive=drive,
    template_g=template_g,
    template_e=template_e,
    template_diff=template_diff,
    threshold=threshold,
    avg_traj_g=avg_traj_g,
    avg_traj_e=avg_traj_e,
)
