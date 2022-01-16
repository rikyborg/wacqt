import warnings

import matplotlib.pyplot as plt
import numpy as np
# from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.linalg import solve
from scipy.optimize import root

w0 = 2 * np.pi * 1.0
kappa = 0.01
wd = w0

fs = 20
w0_2 = w0 * w0
wd_2 = wd * wd
kappa_2 = kappa * kappa
w_hom = w0 * np.sqrt(1.0 - kappa_2 / w0_2 / 4.0)

# in the analytical solution, we assume that the system is underdamped
assert kappa_2 < 4 * w0_2


def ivp_fun(t, y, drive):
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


def ivp_jac(t, y, drive):
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


def hom_coeff(t0, y0, drive):
    """ Find the coefficients for the homogeneous solution.

    Parameters
    ----------
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    (a, b) : tuple of float
        the homogeneous solution is
        np.exp(-(kappa * t) / 2) * (a * np.cos(w_hom * t) - b * np.sin(w_hom * t))
    """
    # solve Ax=b
    A = np.array([
        [
            +np.exp(-(kappa * t0) / 2) * np.cos(w_hom * t0),
            -np.exp(-(kappa * t0) / 2) * np.sin(w_hom * t0),
        ],
        [
            0.5 * np.exp(-(kappa * t0) / 2) *
            (-kappa * np.cos(w_hom * t0) - 2 * w_hom * np.sin(w_hom * t0)),
            0.5 * np.exp(-(kappa * t0) / 2) *
            (-2 * w_hom * np.cos(w_hom * t0) + kappa * np.sin(w_hom * t0)),
        ],
    ])
    b = np.array([
        y0[0] - part_x(t0, drive),
        y0[1] - part_xdot(t0, drive),
    ])
    return solve(A, b)


def hom_x(t, t0, y0, drive):
    """ The homogeneous solution at time `t`.

    Parameters
    ----------
    t : float or array of float
        time
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    float or array of float
    """
    a, b = hom_coeff(t0, y0, drive)
    return np.exp(
        -(kappa * t) / 2) * (a * np.cos(w_hom * t) - b * np.sin(w_hom * t))


def hom_xdot(t, t0, y0, drive):
    """ The derivative of the homogeneous solution at time `t`.

    Parameters
    ----------
    t : float or array of float
        time
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    float or array of float
    """
    a, b = hom_coeff(t0, y0, drive)
    return 0.5 * np.exp(
        -(kappa * t) / 2) * (np.sin(w_hom * t) *
                             (b * kappa - 2 * a * w_hom) - np.cos(w_hom * t) *
                             (a * kappa + 2 * b * w_hom))


def matrix_particular():
    return np.array([
        [
            (w0_2 - wd_2) / ((w0_2 - wd_2)**2 + wd_2 * kappa_2),
            kappa * wd / ((w0_2 - wd_2)**2 + wd_2 * kappa_2),
        ],
        [
            -kappa * wd / ((w0_2 - wd_2)**2 + wd_2 * kappa_2),
            (w0_2 - wd_2) / ((w0_2 - wd_2)**2 + wd_2 * kappa_2),
        ],
    ])


def part_coeff(drive):
    """ Find the coefficients for the particular solution given a drive.

    Parameters
    ----------
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    (c, d) : tuple of float
        the particular solution is c * np.cos(wd * t) - d * np.sin(wd * t)
    """
    m = matrix_particular()
    return (
        drive[0] * m[0][0] + drive[1] * m[0][1],  # c
        drive[0] * m[1][0] + drive[1] * m[1][1],  # d
    )


def part_x(t, drive):
    """ The particular solution at time `t` given a drive.

    Parameters
    ----------
    t : float or array of float
        time
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    float or array of float
    """
    c, d = part_coeff(drive)
    return c * np.cos(wd * t) - d * np.sin(wd * t)


def part_xdot(t, drive):
    """ The derivative of the particular solution at time `t` given a drive.

    Parameters
    ----------
    t : float or array of float
        time
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    float or array of float
    """
    c, d = part_coeff(drive)
    return -c * wd * np.sin(wd * t) - d * wd * np.cos(wd * t)


def an_x(t, t0, y0, drive):
    """ Analytical solution.

    Parameters
    ----------
    t : float or array of float
        time
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    float or array of float
    """
    x_part = part_x(t, drive)
    x_hom = hom_x(t, t0, y0, drive)
    return x_hom + x_part


def an_xdot(t, t0, y0, drive):
    """ Derivative of the analytical solution.

    Parameters
    ----------
    t : float or array of float
        time
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    float or array of float
    """
    xdot_part = part_xdot(t, drive)
    xdot_hom = hom_xdot(t, t0, y0, drive)
    return xdot_hom + xdot_part


def analytical(t, t0, y0, drive):
    """ Convenience function to return the analytical solution in an odeint-like format.

    Parameters
    ----------
    t : array of float
        time
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    drive : 2-array of float
        drive I and Q amplitudes

    Returns
    -------
    array of float with shape (len(t), 2)
    """
    sol = np.empty((len(t), 2), np.float64)
    sol[:, 0] = an_x(t, t0, y0, drive)
    sol[:, 1] = an_xdot(t, t0, y0, drive)
    return sol


def _find_d_ss(t0, y0):
    """ Find the drive amplitudes to produce a steady-state solution.

    Parameters
    ----------
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`

    Returns
    -------
    (float, float)
        the drive I and Q amplitudes
    """
    # NOTE
    # this version could work for a nonlinear x_part and xdot_part
    # but changes to the jac are needed
    def fun(drive):
        x_part = part_x(t0, drive)
        xdot_part = part_xdot(t0, drive)
        return np.array([
            x_part - y0[0],
            xdot_part - y0[1],
        ])

    def jac(drive):
        return np.array([
            [
                part_x(t0, [1.0, 0.0]),
                part_x(t0, [0.0, 1.0]),
            ],
            [
                part_xdot(t0, [1.0, 0.0]),
                part_xdot(t0, [0.0, 1.0]),
            ],
        ])

    sol = root(fun, [0, 0], jac=jac)
    if not sol.success:
        print(sol)
        warnings.warn(f"Failed with message: {sol.message}", UserWarning)
    return sol.x

def find_d_ss(t0, y0):
    """ Find the drive amplitudes to produce a steady-state solution.

    Parameters
    ----------
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`

    Returns
    -------
    (float, float)
        the drive I and Q amplitudes

    Notes
    -----
    Assumes part_x and part_xdot are linear in drive!
    """
    # NOTE: assumes part_x and part_xdot are linear in drive!
    A = np.array([
            [
                part_x(t0, [1.0, 0.0]),
                part_x(t0, [0.0, 1.0]),
            ],
            [
                part_xdot(t0, [1.0, 0.0]),
                part_xdot(t0, [0.0, 1.0]),
            ],
        ])
    b = y0
    return solve(A, b)


def _find_d_zero(t0, y0, t1, y1):
    """ Find the drive amplitudes to reach a given state.

    Parameters
    ----------
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    t1 : float
        time for target condition `y1`
    y1 : 2-array of float
        target state vector `y(t1)`

    Returns
    -------
    (float, float)
        the drive I and Q amplitudes
    """
    # NOTE
    # this version could work for a nonlinear system
    def fun(drive):
        x_an = an_x(t1, t0, y0, drive)
        xdot_an = an_xdot(t1, t0, y0, drive)
        return np.array([
            x_an - y1[0],
            xdot_an - y1[1],
        ])

    sol = root(fun, [0, 0])
    if not sol.success:
        print(sol)
        warnings.warn(f"Failed with message: {sol.message}", UserWarning)
    return sol.x


def find_d_zero(t0, y0, t1, y1):
    """ Find the drive amplitudes to reach a given state.

    Parameters
    ----------
    t0 : float
        time for initial condition `y0`
    y0 : 2-array of float
        initial state vector `y(t0)`
    t1 : float
        time for target condition `y1`
    y1 : 2-array of float
        target state vector `y(t1)`

    Returns
    -------
    (float, float)
        the drive I and Q amplitudes

    Notes
    -----
    Assumes the system is linear in drive!
    """
    # NOTE: assumes the system is linear in drive!
    y1_nodrive = np.array([an_x(t1, t0, y0, [0.0, 0.0]), an_xdot(t1, t0, y0, [0.0, 0.0])])
    A = np.array([
            [
                an_x(t1, t0, y0, [1.0, 0.0]) - y1_nodrive[0],
                an_x(t1, t0, y0, [0.0, 1.0]) - y1_nodrive[0],
            ],
            [
                an_xdot(t1, t0, y0, [1.0, 0.0]) - y1_nodrive[1],
                an_xdot(t1, t0, y0, [0.0, 1.0]) - y1_nodrive[1],
            ],
        ])
    b = y1 - y1_nodrive
    return solve(A, b)


def make_t_arr(t0, t1, fs):
    dt = 1 / fs
    n0 = int(round(t0 * fs))
    n1 = int(round(t1 * fs))
    n_arr = np.arange(n0, n1 + 1)
    return dt * n_arr


# first segment: kick
y0 = np.array([0.0, 0.0])
t0 = 0.0
t1 = 1 / kappa
d01 = [1.0, 0.0]
t01_arr = make_t_arr(t0, t1, fs)
sol01 = odeint(ivp_fun, y0, t01_arr, args=(d01, ), Dfun=ivp_jac, tfirst=True)
an01 = analytical(t01_arr, t0, y0, d01)

# second segment: flat
y1 = sol01[-1]
y1_an = an01[-1]
t2 = t1 + 1 / kappa
d12 = find_d_ss(t1, y1_an)
t12_arr = make_t_arr(t1, t2, fs)
sol12 = odeint(ivp_fun, y1, t12_arr, args=(d12, ), Dfun=ivp_jac, tfirst=True)
an12 = analytical(t12_arr, t1, y1_an, d12)

# third segment: reset
y2 = sol12[-1]
y2_an = an12[-1]
t3 = t2 + 0.5 / kappa
d23 = find_d_zero(t2, y2_an, t3, [0.0, 0.0])
t23_arr = make_t_arr(t2, t3, fs)
sol23 = odeint(ivp_fun, y2, t23_arr, args=(d23, ), Dfun=ivp_jac, tfirst=True)
an23 = analytical(t23_arr, t2, y2_an, d23)

# free evolution
y3 = sol23[-1]
y3_an = an23[-1]
t4 = t3 + 1 / kappa
d34 = [0.0, 0.0]  # no drive
t34_arr = make_t_arr(t3, t4, fs)
sol34 = odeint(ivp_fun, y3, t34_arr, args=(d34, ), Dfun=ivp_jac, tfirst=True)
an34 = analytical(t34_arr, t3, y3_an, d34)

# plot position
fig, ax = plt.subplots(tight_layout=True)
ax.plot(t01_arr, sol01[:, 0])
ax.plot(t12_arr, sol12[:, 0])
ax.plot(t23_arr, sol23[:, 0])
ax.plot(t34_arr, sol34[:, 0])
# ax.plot(t01_arr, an01[:, 0], '--')
# ax.plot(t12_arr, an12[:, 0], '--')
# ax.plot(t23_arr, an23[:, 0], '--')
# ax.plot(t34_arr, an34[:, 0], '--')
ax.grid()
fig.show()

# plot velocity
# fig2, ax2 = plt.subplots(tight_layout=True)
# ax2.plot(t01_arr, sol01[:, 1])
# ax2.plot(t12_arr, sol12[:, 1])
# ax2.plot(t23_arr, sol23[:, 1])
# ax2.plot(t01_arr, an01[:, 1], '--')
# ax2.plot(t12_arr, an12[:, 1], '--')
# ax2.plot(t23_arr, an23[:, 1], '--')
# ax2.grid()
# fig2.show()
