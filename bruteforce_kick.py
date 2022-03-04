from __future__ import annotations

import numpy as np
from numpy.linalg import solve

def erf(x):
    duration_k1, duration_k2, duration_r1 = x
    wr = 2 * np.pi * 6.027848e9 * 1e-9  # GHz
    chi = -2 * np.pi * 302.25e3 * 1e-9
    kappa = 2 * np.pi * 455.13e3 * 1e-9

    i2 = 1.0
    q2 = 1.0

    t0 = 0.0
    t1 = t0 + duration_k1
    t2 = t1 + duration_k2
    t3 = t2
    t4 = t3 + duration_r1
    t5 = 1000.0
    # t1 = t0 + 370.0
    # t2 = t1 + 130.0
    # t3 = t2 + 0.0
    # t4 = t3 + 300.0
    # t5 = t4 + 200.0
    # t6 = t5 + 100.0

    wg = wr - chi
    we = wr + chi
    wd = wr

    fs = 60.0
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
        n_arr = np.arange(n0, n1)
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
    (
        a0_g, b0_g, a0_e, b0_e, i0, q0,
        a1_g, b1_g, a1_e, b1_e, i1, q1,
        a3_g, b3_g, a3_e, b3_e, i3, q3,
        a4_g, b4_g, a4_e, b4_e, i4, q4,
     ) = all_sol

    t01_arr = make_t_arr(t0, t1, fs)
    t12_arr = make_t_arr(t1, t2, fs)
    # no flat
    t34_arr = make_t_arr(t3, t4, fs)
    t45_arr = make_t_arr(t4, t5, fs)


    x01_g = a0_g * alpha(t01_arr, wg) + b0_g * beta(t01_arr, wg) + i0 * gamma(t01_arr, wg) + q0 * delta(t01_arr, wg)
    x01_e = a0_e * alpha(t01_arr, we) + b0_e * beta(t01_arr, we) + i0 * gamma(t01_arr, we) + q0 * delta(t01_arr, we)
    x12_g = a1_g * alpha(t12_arr, wg) + b1_g * beta(t12_arr, wg) + i1 * gamma(t12_arr, wg) + q1 * delta(t12_arr, wg)
    x12_e = a1_e * alpha(t12_arr, we) + b1_e * beta(t12_arr, we) + i1 * gamma(t12_arr, we) + q1 * delta(t12_arr, we)
    x34_g = a3_g * alpha(t34_arr, wg) + b3_g * beta(t34_arr, wg) + i3 * gamma(t34_arr, wg) + q3 * delta(t34_arr, wg)
    x34_e = a3_e * alpha(t34_arr, we) + b3_e * beta(t34_arr, we) + i3 * gamma(t34_arr, we) + q3 * delta(t34_arr, we)
    x45_g = a4_g * alpha(t45_arr, wg) + b4_g * beta(t45_arr, wg) + i4 * gamma(t45_arr, wg) + q4 * delta(t45_arr, wg)
    x45_e = a4_e * alpha(t45_arr, we) + b4_e * beta(t45_arr, we) + i4 * gamma(t45_arr, we) + q4 * delta(t45_arr, we)

    v01_g = a0_g * alpha_dot(t01_arr, wg) + b0_g * beta_dot(t01_arr, wg) + i0 * gamma_dot(t01_arr, wg) + q0 * delta_dot(t01_arr, wg)
    v01_e = a0_e * alpha_dot(t01_arr, we) + b0_e * beta_dot(t01_arr, we) + i0 * gamma_dot(t01_arr, we) + q0 * delta_dot(t01_arr, we)
    v12_g = a1_g * alpha_dot(t12_arr, wg) + b1_g * beta_dot(t12_arr, wg) + i1 * gamma_dot(t12_arr, wg) + q1 * delta_dot(t12_arr, wg)
    v12_e = a1_e * alpha_dot(t12_arr, we) + b1_e * beta_dot(t12_arr, we) + i1 * gamma_dot(t12_arr, we) + q1 * delta_dot(t12_arr, we)
    v34_g = a3_g * alpha_dot(t34_arr, wg) + b3_g * beta_dot(t34_arr, wg) + i3 * gamma_dot(t34_arr, wg) + q3 * delta_dot(t34_arr, wg)
    v34_e = a3_e * alpha_dot(t34_arr, we) + b3_e * beta_dot(t34_arr, we) + i3 * gamma_dot(t34_arr, we) + q3 * delta_dot(t34_arr, we)
    v45_g = a4_g * alpha_dot(t45_arr, wg) + b4_g * beta_dot(t45_arr, wg) + i4 * gamma_dot(t45_arr, wg) + q4 * delta_dot(t45_arr, wg)
    v45_e = a4_e * alpha_dot(t45_arr, we) + b4_e * beta_dot(t45_arr, we) + i4 * gamma_dot(t45_arr, we) + q4 * delta_dot(t45_arr, we)

    t_arr = np.r_[
        t01_arr,
        t12_arr,
        t34_arr,
        t45_arr,
    ]
    x_g = np.r_[
        x01_g,
        x12_g,
        x34_g,
        x45_g,
    ]
    x_e = np.r_[
        x01_e,
        x12_e,
        x34_e,
        x45_e,
    ]
    v_g = np.r_[
        v01_g,
        v12_g,
        v34_g,
        v45_g,
    ]
    v_e = np.r_[
        v01_e,
        v12_e,
        v34_e,
        v45_e,
    ]
    v_g /= wd
    v_e /= wd
    sep = np.sqrt((x_g - x_e)**2 + (v_g - v_e)**2)

    d_arr = np.array([i0, q0, i1, q1, i2, q2, i3, q3, i4, q4])
    d_max = np.max(np.abs(d_arr))
    s_norm = np.sum(sep) / d_max / fs
    return s_norm

x_best = None
s_best = 0.0
step = 10
for duration_k1 in range(step, 1_000, step):
    if duration_k1 > 994:
        break
    for duration_k2  in range(step, 1_000, step):
        if duration_k1 + duration_k2 > 996:
            break
        for duration_r1  in range(step, 1_000, step):
            if duration_k1 + duration_k2 + duration_r1 > 998:
                break
            x = [duration_k1, duration_k2, duration_r1]
            s = erf(x)
            print(f"[{x[0]:03d} - {x[1]:03d} - {x[2]:03d}] \t {s:06.1f} \t {s_best:06.1f}")
            if s > s_best:
                s_best = s
                x_best = x
