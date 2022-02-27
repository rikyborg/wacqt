from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
# from scipy.linalg import solve

# w0 = 2 * np.pi * 1.0
wr = 2 * np.pi * 1.0
kappa = 0.1
chi = -0.05
wg = wr - chi
we = wr + chi
wd = wr

fs = 20
wg_2 = wg * wg
we_2 = we * we
wd_2 = wd * wd
kappa_2 = kappa * kappa

# in the analytical solution, we assume that the system is underdamped
assert kappa_2 < 4 * wg_2
assert kappa_2 < 4 * we_2


def tilde(w0: float) -> float:
    return w0 * np.sqrt(1 - (kappa / (2 * w0))**2)


def alpha(t: float, w0: float) -> float:
    return np.exp(-kappa * t / 2) * np.cos(tilde(w0) * t)


def beta(t: float, w0: float) -> float:
    return -np.exp(-kappa * t / 2) * np.sin(tilde(w0) * t)


def gamma(t: float, w0: float) -> float:
    d = (w0**2 - wd**2)**2 + kappa**2 * wd**2
    nc = w0**2 - wd**2
    ns = kappa * wd
    return nc / d * np.cos(wd * t) + ns / d * np.sin(wd * t)


def delta(t: float, w0: float) -> float:
    d = (w0**2 - wd**2)**2 + kappa**2 * wd**2
    nc = kappa * wd
    ns = w0**2 - wd**2
    return nc / d * np.cos(wd * t) - ns / d * np.sin(wd * t)


def alpha_dot(t: float, w0: float) -> float:
    return alpha(t, w0) * (kappa**2 / 4 - tilde(w0)**2) - beta(t, w0) * kappa * tilde(w0)


def beta_dot(t: float, w0: float) -> float:
    return alpha(t, w0) * kappa * tilde(w0) - beta(t, w0) * (tilde(w0)**2 - kappa**2 / 4)


def gamma_dot(t: float, w0: float) -> float:
    return wd * delta(t, w0)


def delta_dot(t: float, w0: float) -> float:
    return -wd * gamma(t, w0)


def solve_all(i2: float, q2: float, t0: float, t1: float, t2: float):
    # yapf: disable
    a = np.array([
        [alpha(t0, wg),      beta(t0, wg),      0,                  0,                 gamma(t0, wg),      delta(t0, wg),      0,                 0,                0,                 0,                0,                 0                ],
        [alpha_dot(t0, wg),  beta_dot(t0, wg),  0,                  0,                 gamma_dot(t0, wg),  delta_dot(t0, wg),  0,                 0,                0,                 0,                0,                 0                ],
        [0,                  0,                 alpha(t0, we),      beta(t0, we),      gamma(t0, we),      delta(t0, we),      0,                 0,                0,                 0,                0,                 0                ],
        [0,                  0,                 alpha_dot(t0, we),  beta_dot(t0, we),  gamma_dot(t0, we),  delta_dot(t0, we),  0,                 0,                0,                 0,                0,                 0                ],
        [-alpha(t1, wg),     -beta(t1, wg),     0,                  0,                 -gamma(t1, wg),     -delta(t1, wg),     alpha(t1, wg),     beta(t1, wg),     0,                 0,                gamma(t1, wg),     delta(t1, wg)    ],
        [-alpha_dot(t1, wg), -beta_dot(t1, wg), 0,                  0,                 -gamma_dot(t1, wg), -delta_dot(t1, wg), alpha_dot(t1, wg), beta_dot(t1, wg), 0,                 0,                gamma_dot(t1, wg), delta_dot(t1, wg)],
        [0,                  0,                 -alpha(t1, we),     -beta(t1, we),     -gamma(t1, we),     -delta(t1, we),     0,                 0,                alpha(t1, we),     beta(t1, we),     gamma(t1, we),     delta(t1, we),   ],
        [0,                  0,                 -alpha_dot(t1, we), -beta_dot(t1, we), -gamma_dot(t1, we), -delta_dot(t1, we), 0,                 0,                alpha_dot(t1, we), beta_dot(t1, we), gamma_dot(t1, we), delta_dot(t1, we)],
        [0,                  0,                 0,                  0,                 0,                  0,                  alpha(t2, wg),     beta(t2, wg),     0,                 0,                gamma(t2, wg),     delta(t2, wg)    ],
        [0,                  0,                 0,                  0,                 0,                  0,                  alpha_dot(t2, wg), beta_dot(t2, wg), 0,                 0,                gamma_dot(t2, wg), delta_dot(t2, wg)],
        [0,                  0,                 0,                  0,                 0,                  0,                  0,                 0,                alpha(t2, we),     beta(t2, we),     gamma(t2, we),     delta(t2, we)    ],
        [0,                  0,                 0,                  0,                 0,                  0,                  0,                 0,                alpha_dot(t2, we), beta_dot(t2, we), gamma_dot(t2, we), delta_dot(t2, we)],
    ])
    # yapf: enable

    b = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        gamma(t2, wg) * i2 + delta(t2, wg) * q2,
        gamma_dot(t2, wg) * i2 + delta_dot(t2, wg) * q2,
        gamma(t2, we) * i2 + delta(t2, we) * q2,
        gamma_dot(t2, we) * i2 + delta_dot(t2, we) * q2,
    ])

    return np.linalg.solve(a, b)


def ivp_fun(t, y, drive, w0_2):
    """ The ODE function ydot = fun(t, y).

    Parameters
    ----------
    t : float
        time
    y : 2-array of float
        state vector
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    2-array of float
        the derivative of the state vector
    """
    drive_term = drive[0] * np.cos(wd * t) - drive[1] * np.sin(wd * t)
    return np.array([
        y[1],
        -kappa * y[1] - w0_2 * y[0] + drive_term,
    ])


def ivp_jac(t, y, drive, w0_2):
    """ The Jacobian matrix for the ODE.

    Parameters
    ----------
    t : float
        time
    y : 2-array of float
        state vector
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    2x2-matrix of float
        the Jacobian
    """
    return np.array([
        [0.0, 1.0],
        [-w0_2, -kappa],
    ])


def make_t_arr(t0, t1, fs):
    dt = 1 / fs
    n0 = int(round(t0 * fs))
    n1 = int(round(t1 * fs))
    n_arr = np.arange(n0, n1 + 1)
    return dt * n_arr


t0 = 0.0
t1 = t0 + 1 / kappa
t2 = t1 + 1 / kappa
t3 = t2 + 3.0 / kappa
i2 = 1.0
q2 = 0.0
all_sol = solve_all(
    i2=i2,
    q2=q2,
    t0=t0,
    t1=t1,
    t2=t2,
)
i0, q0 = all_sol[4], all_sol[5]
i1, q1 = all_sol[10], all_sol[11]

# first segment: kick 1
y0 = np.array([0.0, 0.0])
t0 = 0.0
d01 = [i0, q0]
t01_arr = make_t_arr(t0, t1, fs)
sol01_g = odeint(ivp_fun, y0, t01_arr, args=(d01, wg_2), Dfun=ivp_jac, tfirst=True)
sol01_e = odeint(ivp_fun, y0, t01_arr, args=(d01, we_2), Dfun=ivp_jac, tfirst=True)

# second segment: kick 2
y1_g = sol01_g[-1]
y1_e = sol01_e[-1]
d12 = [i1, q1]
t12_arr = make_t_arr(t1, t2, fs)
sol12_g = odeint(ivp_fun, y1_g, t12_arr, args=(d12, wg_2), Dfun=ivp_jac, tfirst=True)
sol12_e = odeint(ivp_fun, y1_e, t12_arr, args=(d12, we_2), Dfun=ivp_jac, tfirst=True)

# third segment: flat
y2_g = sol12_g[-1]
y2_e = sol12_e[-1]
d23 = [i2, q2]
t23_arr = make_t_arr(t2, t3, fs)
sol23_g = odeint(ivp_fun, y2_g, t23_arr, args=(d23, wg_2), Dfun=ivp_jac, tfirst=True)
sol23_e = odeint(ivp_fun, y2_e, t23_arr, args=(d23, we_2), Dfun=ivp_jac, tfirst=True)

# plot position
fig, ax = plt.subplots(tight_layout=True)
ax.plot(t01_arr, sol01_g[:, 0], c="tab:blue", label="|g>")
ax.plot(t01_arr, sol01_e[:, 0], c="tab:orange", label="|e>")
ax.plot(t12_arr, sol12_g[:, 0], c="tab:blue")
ax.plot(t12_arr, sol12_e[:, 0], c="tab:orange")
ax.plot(t23_arr, sol23_g[:, 0], c="tab:blue")
ax.plot(t23_arr, sol23_e[:, 0], c="tab:orange")
ax.axvline(t0, ls='-', c="k", alpha=0.5)
ax.axvline(t1, ls='-', c="k", alpha=0.5)
ax.axvline(t2, ls='-', c="k", alpha=0.5)
ax.axvline(t3, ls='-', c="k", alpha=0.5)
ax.grid()
ax.legend(ncol=2)
ax.set_xlabel("Time")
ax.set_ylabel("Position")
fig.show()
