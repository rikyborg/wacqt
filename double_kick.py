from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from numpy.linalg import solve
# from scipy.linalg import solve

wr = 2 * np.pi * 6.027848e9 * 1e-9  # GHz
chi = -2 * np.pi * 302.25e3 * 1e-9
kappa = 2 * np.pi * 455.13e3 * 1e-9

i2 = 1.0
q2 = 1.0

t0 = 0.0
t1 = t0 + 520.0
t2 = t1 + 120.0
t3 = t2 + 0.0
t4 = t3 + 340.0
t5 = t4 + 220.0
t6 = t5 + 100.0

wg = wr - chi
we = wr + chi
wd = wr

fs = 120
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
    return -kappa / 2 * alpha(t, w0) + tilde(w0) * beta(t, w0)


def beta_dot(t: float, w0: float) -> float:
    return -tilde(w0) * alpha(t, w0) - kappa / 2 * beta(t, w0)


def gamma_dot(t: float, w0: float) -> float:
    return wd * delta(t, w0)


def delta_dot(t: float, w0: float) -> float:
    return -wd * gamma(t, w0)


def solve_all(i2: float, q2: float, t0: float, t1: float, t2: float, t3: float,
              t4: float, t5: float):
    # yapf: disable
    a_kick = np.array([
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

    b_kick = np.array([
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

    sol_kick = solve(a_kick, b_kick)

    # yapf: disable
    a_reset = np.array([
        [alpha(t3, wg),      beta(t3, wg),      0,                  0,                 gamma(t3, wg),      delta(t3, wg),      0,                 0,                0,                 0,                0,                 0                ],
        [alpha_dot(t3, wg),  beta_dot(t3, wg),  0,                  0,                 gamma_dot(t3, wg),  delta_dot(t3, wg),  0,                 0,                0,                 0,                0,                 0                ],
        [0,                  0,                 alpha(t3, we),      beta(t3, we),      gamma(t3, we),      delta(t3, we),      0,                 0,                0,                 0,                0,                 0                ],
        [0,                  0,                 alpha_dot(t3, we),  beta_dot(t3, we),  gamma_dot(t3, we),  delta_dot(t3, we),  0,                 0,                0,                 0,                0,                 0                ],
        [-alpha(t4, wg),     -beta(t4, wg),     0,                  0,                 -gamma(t4, wg),     -delta(t4, wg),     alpha(t4, wg),     beta(t4, wg),     0,                 0,                gamma(t4, wg),     delta(t4, wg)    ],
        [-alpha_dot(t4, wg), -beta_dot(t4, wg), 0,                  0,                 -gamma_dot(t4, wg), -delta_dot(t4, wg), alpha_dot(t4, wg), beta_dot(t4, wg), 0,                 0,                gamma_dot(t4, wg), delta_dot(t4, wg)],
        [0,                  0,                 -alpha(t4, we),     -beta(t4, we),     -gamma(t4, we),     -delta(t4, we),     0,                 0,                alpha(t4, we),     beta(t4, we),     gamma(t4, we),     delta(t4, we),   ],
        [0,                  0,                 -alpha_dot(t4, we), -beta_dot(t4, we), -gamma_dot(t4, we), -delta_dot(t4, we), 0,                 0,                alpha_dot(t4, we), beta_dot(t4, we), gamma_dot(t4, we), delta_dot(t4, we)],
        [0,                  0,                 0,                  0,                 0,                  0,                  alpha(t5, wg),     beta(t5, wg),     0,                 0,                gamma(t5, wg),     delta(t5, wg)    ],
        [0,                  0,                 0,                  0,                 0,                  0,                  alpha_dot(t5, wg), beta_dot(t5, wg), 0,                 0,                gamma_dot(t5, wg), delta_dot(t5, wg)],
        [0,                  0,                 0,                  0,                 0,                  0,                  0,                 0,                alpha(t5, we),     beta(t5, we),     gamma(t5, we),     delta(t5, we)    ],
        [0,                  0,                 0,                  0,                 0,                  0,                  0,                 0,                alpha_dot(t5, we), beta_dot(t5, we), gamma_dot(t5, we), delta_dot(t5, we)],
    ])
    # yapf: enable

    b_reset = np.array([
        gamma(t3, wg) * i2 + delta(t3, wg) * q2,
        gamma_dot(t3, wg) * i2 + delta_dot(t3, wg) * q2,
        gamma(t3, we) * i2 + delta(t3, we) * q2,
        gamma_dot(t3, we) * i2 + delta_dot(t3, we) * q2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])

    sol_reset = solve(a_reset, b_reset)

    return np.r_[sol_kick, sol_reset]


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


all_sol = solve_all(
    i2=i2,
    q2=q2,
    t0=t0,
    t1=t1,
    t2=t2,
    t3=t3,
    t4=t4,
    t5=t5,
)
i0, q0 = all_sol[4], all_sol[5]
i1, q1 = all_sol[10], all_sol[11]
i3, q3 = all_sol[16], all_sol[17]
i4, q4 = all_sol[22], all_sol[23]

# first segment: kick 1
y0 = np.array([0.0, 0.0])
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
if t3 > t2:
    t23_arr = make_t_arr(t2, t3, fs)
    sol23_g = odeint(ivp_fun, y2_g, t23_arr, args=(d23, wg_2), Dfun=ivp_jac, tfirst=True)
    sol23_e = odeint(ivp_fun, y2_e, t23_arr, args=(d23, we_2), Dfun=ivp_jac, tfirst=True)
else:
    assert t2 == t3
    t23_arr = np.array([t2, t3])
    sol23_g = np.vstack((y2_g, y2_g))
    sol23_e = np.vstack((y2_e, y2_e))

# fourth segment: reset 1
y3_g = sol23_g[-1]
y3_e = sol23_e[-1]
d34 = [i3, q3]
t34_arr = make_t_arr(t3, t4, fs)
sol34_g = odeint(ivp_fun, y3_g, t34_arr, args=(d34, wg_2), Dfun=ivp_jac, tfirst=True)
sol34_e = odeint(ivp_fun, y3_e, t34_arr, args=(d34, we_2), Dfun=ivp_jac, tfirst=True)

# fifth segment: reset 3
y4_g = sol34_g[-1]
y4_e = sol34_e[-1]
d45 = [i4, q4]
t45_arr = make_t_arr(t4, t5, fs)
sol45_g = odeint(ivp_fun, y4_g, t45_arr, args=(d45, wg_2), Dfun=ivp_jac, tfirst=True)
sol45_e = odeint(ivp_fun, y4_e, t45_arr, args=(d45, we_2), Dfun=ivp_jac, tfirst=True)

# last segment: free evolution
y5_g = sol45_g[-1]
y5_e = sol45_e[-1]
d56 = [0.0, 0.0]
t56_arr = make_t_arr(t5, t6, fs)
sol56_g = odeint(ivp_fun, y5_g, t56_arr, args=(d56, wg_2), Dfun=ivp_jac, tfirst=True)
sol56_e = odeint(ivp_fun, y5_e, t56_arr, args=(d56, we_2), Dfun=ivp_jac, tfirst=True)

# complex
sol01_g = np.frombuffer(sol01_g, np.complex128)
sol01_e = np.frombuffer(sol01_e, np.complex128)
sol12_g = np.frombuffer(sol12_g, np.complex128)
sol12_e = np.frombuffer(sol12_e, np.complex128)
sol23_g = np.frombuffer(sol23_g, np.complex128)
sol23_e = np.frombuffer(sol23_e, np.complex128)
sol34_g = np.frombuffer(sol34_g, np.complex128)
sol34_e = np.frombuffer(sol34_e, np.complex128)
sol45_g = np.frombuffer(sol45_g, np.complex128)
sol45_e = np.frombuffer(sol45_e, np.complex128)
sol56_g = np.frombuffer(sol56_g, np.complex128)
sol56_e = np.frombuffer(sol56_e, np.complex128)

# units
sol01_g.imag /= wd
sol01_e.imag /= wd
sol12_g.imag /= wd
sol12_e.imag /= wd
sol23_g.imag /= wd
sol23_e.imag /= wd
sol34_g.imag /= wd
sol34_e.imag /= wd
sol45_g.imag /= wd
sol45_e.imag /= wd
sol56_g.imag /= wd
sol56_e.imag /= wd

# plot position
fig, ax = plt.subplots(tight_layout=True)
ax.plot(t01_arr, sol01_g.real, c="tab:blue", label="|g>")
ax.plot(t01_arr, sol01_e.real, c="tab:orange", label="|e>")
ax.plot(t12_arr, sol12_g.real, c="tab:blue")
ax.plot(t12_arr, sol12_e.real, c="tab:orange")
ax.plot(t23_arr, sol23_g.real, c="tab:blue")
ax.plot(t23_arr, sol23_e.real, c="tab:orange")
ax.plot(t34_arr, sol34_g.real, c="tab:blue")
ax.plot(t34_arr, sol34_e.real, c="tab:orange")
ax.plot(t45_arr, sol45_g.real, c="tab:blue")
ax.plot(t45_arr, sol45_e.real, c="tab:orange")
ax.plot(t56_arr, sol56_g.real, c="tab:blue")
ax.plot(t56_arr, sol56_e.real, c="tab:orange")
ax.axvline(t0, ls='-', c="k", alpha=0.5)
ax.axvline(t1, ls='-', c="k", alpha=0.5)
ax.axvline(t2, ls='-', c="k", alpha=0.5)
ax.axvline(t3, ls='-', c="k", alpha=0.5)
ax.axvline(t4, ls='-', c="k", alpha=0.5)
ax.axvline(t5, ls='-', c="k", alpha=0.5)
ax.grid()
ax.legend(ncol=2)
ax.set_xlabel("Time")
ax.set_ylabel("Position")
fig.show()

# plot energy
n01_g = np.abs(sol01_g)**2 / 2
n01_e = np.abs(sol01_e)**2 / 2
n12_g = np.abs(sol12_g)**2 / 2
n12_e = np.abs(sol12_e)**2 / 2
n23_g = np.abs(sol23_g)**2 / 2
n23_e = np.abs(sol23_e)**2 / 2
n34_g = np.abs(sol34_g)**2 / 2
n34_e = np.abs(sol34_e)**2 / 2
n45_g = np.abs(sol45_g)**2 / 2
n45_e = np.abs(sol45_e)**2 / 2
n56_g = np.abs(sol56_g)**2 / 2
n56_e = np.abs(sol56_e)**2 / 2

fig2, ax2 = plt.subplots(tight_layout=True)
ax2.plot(t01_arr, n01_g, c="tab:blue", label="|g>")
ax2.plot(t01_arr, n01_e, c="tab:orange", label="|e>")
ax2.plot(t12_arr, n12_g, c="tab:blue")
ax2.plot(t12_arr, n12_e, c="tab:orange")
ax2.plot(t23_arr, n23_g, c="tab:blue")
ax2.plot(t23_arr, n23_e, c="tab:orange")
ax2.plot(t34_arr, n34_g, c="tab:blue")
ax2.plot(t34_arr, n34_e, c="tab:orange")
ax2.plot(t45_arr, n45_g, c="tab:blue")
ax2.plot(t45_arr, n45_e, c="tab:orange")
ax2.plot(t56_arr, n56_g, c="tab:blue")
ax2.plot(t56_arr, n56_e, c="tab:orange")
ax2.axvline(t0, ls='-', c="k", alpha=0.5)
ax2.axvline(t1, ls='-', c="k", alpha=0.5)
ax2.axvline(t2, ls='-', c="k", alpha=0.5)
ax2.axvline(t3, ls='-', c="k", alpha=0.5)
ax2.axvline(t4, ls='-', c="k", alpha=0.5)
ax2.axvline(t5, ls='-', c="k", alpha=0.5)
ax2.grid()
ax2.legend(ncol=2)
ax2.set_xlabel("Time")
ax2.set_ylabel("Energy")
fig2.show()

# separation
s01 = np.abs(sol01_g - sol01_e)
s12 = np.abs(sol12_g - sol12_e)
s23 = np.abs(sol23_g - sol23_e)
s34 = np.abs(sol34_g - sol34_e)
s45 = np.abs(sol45_g - sol45_e)
s56 = np.abs(sol56_g - sol56_e)
s_all = np.r_[s01[:-1], s12[:-1], s23[:-1], s34[:-1], s45[:-1], s56[:-1]]

d_arr = np.array([i0, q0, i1, q1, i2, q2, i3, q3, i4, q4])
d_max = np.max(np.abs(d_arr))
print(np.sum(s_all) / d_max / fs)

fig3, ax3 = plt.subplots(tight_layout=True)
ax3.plot(t01_arr, s01, c="tab:blue")
ax3.plot(t12_arr, s12, c="tab:blue")
ax3.plot(t23_arr, s23, c="tab:blue")
ax3.plot(t34_arr, s34, c="tab:blue")
ax3.plot(t45_arr, s45, c="tab:blue")
ax3.plot(t56_arr, s56, c="tab:blue")
ax3.plot(t01_arr, s01 / d_max, c="tab:green")
ax3.plot(t12_arr, s12 / d_max, c="tab:green")
ax3.plot(t23_arr, s23 / d_max, c="tab:green")
ax3.plot(t34_arr, s34 / d_max, c="tab:green")
ax3.plot(t45_arr, s45 / d_max, c="tab:green")
ax3.plot(t56_arr, s56 / d_max, c="tab:green")
ax3.axvline(t0, ls='-', c="k", alpha=0.5)
ax3.axvline(t1, ls='-', c="k", alpha=0.5)
ax3.axvline(t2, ls='-', c="k", alpha=0.5)
ax3.axvline(t3, ls='-', c="k", alpha=0.5)
ax3.axvline(t4, ls='-', c="k", alpha=0.5)
ax3.axvline(t5, ls='-', c="k", alpha=0.5)
ax3.grid()
ax3.set_xlabel("Time")
ax3.set_ylabel("Separation")
fig3.show()
