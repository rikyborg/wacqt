# Based on: /home/riccardo/Documents/PhD/noise_variable_step_integrator/cvode_noise_alex.py
from __future__ import division, print_function

import ctypes
import os

import numpy as np
import numpy.ctypeslib as npct
from scipy.constants import Boltzmann

ID_LINEAR = 0
ID_DUFFING = 1

c_double = ctypes.c_double
c_double_p = ctypes.POINTER(ctypes.c_double)


class SimPara(ctypes.Structure):
    """ Structure used to pass the relevant simulation parameters
    to the C code. The fields here MUST match the fields of the
    SimPara structure in the C code: same name, type and ORDER!
    """
    _fields_ = [
        # Circuit
        ('Cl', c_double),  # F
        ('Cr', c_double),  # F
        ('R1', c_double),  # ohm
        ('L1', c_double),  # H
        ('C1', c_double),  # F
        ('R2', c_double),  # ohm
        ('L2', c_double),  # H
        ('C2', c_double),  # F

        # Drive
        ('nr_drives', ctypes.c_int),
        ('w_arr', c_double_p),  # rad/s
        ('A_arr', c_double_p),  # V
        ('P_arr', c_double_p),  # rad

        # Select system
        ('sys_id', ctypes.c_int),  # ID_LINEAR or ID_DUFFING

        # Duffing: V/L * (1. - duff * V*V)
        ('duff', c_double),  # V^-2

        # Thermal noise
        ('add_thermal_noise', ctypes.c_int),  # bool
        ('dt_noise', c_double),  # s
        ('noise1_array', c_double_p),  # V
        ('noise2_array', c_double_p),  # V
    ]


class SimulationParameters(object):
    """ Container class used for saving the simulation parameters.

    Attributes:
        para (SimPara): the C structure to be passed to the cvode simulator.
    """

    def __init__(
        self,
        Cl, Cr,
        R1, L1, C1,
        R2, L2, C2,
        w_arr, A_arr, P_arr,
    ):
        """
        Args:
            create_noise (bool, optional): if True, creates the array with the
                random thermal noise force, and the C structure for the simulator.
        """
        self.Cl = Cl
        self.Cr = Cr
        self.C1 = C1
        self.L1 = L1
        self.R1 = R1
        self.C2 = C2
        self.L2 = L2
        self.R2 = R2
        self.Csum1 = self.C1 + self.Cl + self.Cr
        self.Csum2 = self.C2 + self.Cr

        self.w01 = np.sqrt(1. / (self.C1 * self.L1))  # rad/s, bare cavity 1 resonance frequency
        self.f01 = self.w01 / 2. / np.pi  # Hz, bare cavity 1 resonance frequency
        self.Q1 = self.R1 * np.sqrt(self.C1 / self.L1)  # bare cavity 1 quality factor
        self.w02 = np.sqrt(1. / (self.C2 * self.L2))  # rad/s, bare cavity 2 resonance frequency
        self.f02 = self.w02 / 2. / np.pi  # Hz, bare cavity 2 resonance frequency
        self.Q2 = self.R2 * np.sqrt(self.C2 / self.L2)  # bare cavity 2 quality factor

        self.Nbeats = 1  # nr of windows (periods, beats) to simulate

        self.fs = 50e9  # Hz
        self.df = 50e6  # Hz
        self.dw = 2. * np.pi * self.df
        self.df = self.dw / (2. * np.pi)
        self.ns = int(round(self.fs / self.df))  # nr of samples in one windows (period, beat)
        self.dt = 1. / self.fs
        self.T = 1. / self.df
        self.ws = 2. * np.pi * self.fs

        self.w_arr = np.atleast_1d(w_arr).copy()
        assert len(self.w_arr.shape) == 1
        self.nr_drives = len(self.w_arr)
        self.A_arr = np.atleast_1d(A_arr).copy()
        assert len(self.A_arr) == self.nr_drives
        self.P_arr = np.atleast_1d(P_arr).copy()
        assert len(self.P_arr) == self.nr_drives
        self.n_arr = np.round(self.w_arr / self.dw).astype(np.int64)

        self.add_thermal_noise = False
        self.noise_T1 = 0.  # K
        self.noise_T2 = 0.  # K
        self.noise1_array = None
        self.noise2_array = None
        self.dt_noise = self.dt

        self.para = SimPara()
        self.para.Cl = float(self.Cl)
        self.para.Cr = float(self.Cr)
        self.para.R1 = float(self.R1)
        self.para.L1 = float(self.L1)
        self.para.C1 = float(self.C1)
        self.para.R2 = float(self.R2)
        self.para.L2 = float(self.L2)
        self.para.C2 = float(self.C2)
        self.para.nr_drives = self.nr_drives
        self.para.w_arr = npct.as_ctypes(self.w_arr)
        self.para.A_arr = npct.as_ctypes(self.A_arr)
        self.para.P_arr = npct.as_ctypes(self.P_arr)
        self.para.sys_id = ID_LINEAR
        self.para.duff = 0.
        self.para.add_thermal_noise = self.add_thermal_noise
        self.para.dt_noise = self.dt_noise
        self.para.noise1_array = c_double_p()
        self.para.noise2_array = c_double_p()

    def set_df(self, df):
        self.set_dw(2. * np.pi * df)

    def set_dw(self, dw):
        self.dw = float(dw)
        self.df = self.dw / (2. * np.pi)
        self.ns = int(round(self.fs / self.df))
        self.T = 1. / self.df
        self.n_arr = np.round(self.w_arr / self.dw).astype(np.int64)

    def set_drive_frequencies(self, f_arr):
        f_arr = np.atleast_1d(f_arr)
        self.set_w_arr(2. * np.pi * f_arr)

    def set_w_arr(self, w_arr):
        w_arr = np.atleast_1d(w_arr).copy()
        assert len(w_arr.shape) == 1
        assert len(w_arr) == self.nr_drives
        self.w_arr = w_arr
        self.n_arr = np.round(self.w_arr / self.dw).astype(np.int64)
        self.para.w_arr = npct.as_ctypes(self.w_arr)

    def set_Nbeats(self, Nbeats):
        self.Nbeats = int(Nbeats)

    def set_duffing(self, duff):
        self.para.duff = float(duff)
        self.para.sys_id = ID_DUFFING

    def set_noise_T(self, T, T2=None):
        if T2 is None:
            T2 = T
        self.noise_T1 = float(T)
        self.noise_T2 = float(T2)

        if self.noise_T1 or self.noise_T2:
            PSDv1_onesided = 4. * Boltzmann * self.noise_T1 * self.R1
            PSDv2_onesided = 4. * Boltzmann * self.noise_T2 * self.R2
            PSDv1_twosided = PSDv1_onesided / 2.
            PSDv2_twosided = PSDv2_onesided / 2.
            self.add_thermal_noise = True
            self.para.add_thermal_noise = True
            self.noise1_array = np.sqrt(PSDv1_twosided) * np.sqrt(self.fs) * np.random.randn(self.Nbeats * self.ns + 2)
            self.noise2_array = np.sqrt(PSDv2_twosided) * np.sqrt(self.fs) * np.random.randn(self.Nbeats * self.ns + 2)
            self.para.noise1_array = npct.as_ctypes(self.noise1_array)
            self.para.noise2_array = npct.as_ctypes(self.noise2_array)
        else:
            self.add_thermal_noise = False
            self.para.add_thermal_noise = False
            self.noise1_array = None
            self.noise2_array = None
            self.para.noise1_array = c_double_p()
            self.para.noise2_array = c_double_p()

    def simulate(self, init=None, out=None):
        if init is None:
            init = np.array([0., 0., 0., 0.])
        if out is None:
            out = np.empty((self.Nbeats * self.ns, 4))

        res = c_lib.integrate_cvode(
            ctypes.byref(self.para),
            0., init,
            1.49012e-8, 1.49012e-8,
            self.Nbeats * self.ns, self.dt,
            out,
        )
        if res:
            print("*** Some error occurred in CVODE. Return flag: {:d}".format(res))
        return out

    def tune(self, f, df, prio_f=False):
        if f and prio_f:
            n = int(round(f / df))
            df = f / n
            ns = int(round(self.fs / df))
            df_out = self.fs / ns
            n = int(round(f / df_out))
            f_out = n * df_out
        else:
            ns = int(round(self.fs / df))
            df_out = self.fs / ns
            n = int(round(f / df_out))
            f_out = n * df_out
        return f_out, df_out

    def tune_w(self, w, dw, prio_w=False):
        if w and prio_w:
            n = int(round(w / dw))
            dw = w / n
            ns = int(round(self.ws / dw))
            dw_out = self.ws / ns
            n = int(round(w / dw_out))
            w_out = n * dw_out
        else:
            ns = int(round(self.ws / dw))
            dw_out = dw * ns
            n = int(round(w / dw_out))
            w_out = n * dw_out
        return w_out, dw_out

    def tf1(self, f):
        s = 1j * 2. * np.pi * f
        Cl = self.Cl
        Cr = self.Cr
        R1 = self.R1
        L1 = self.L1
        C1 = self.C1
        R2 = self.R2
        L2 = self.L2
        C2 = self.C2
        return (Cl * L1 * R1 * s**2 * (R2 + L2 * s + (C2 + Cr) * L2 * R2 * s**2)) / (L1 * s * (R2 + L2 * s + (C2 + Cr) * L2 * R2 * s**2) + R1 * (R2 + L2 * s + ((C1 + Cl + Cr) * L1 + (C2 + Cr) * L2) * R2 * s**2 + (C1 + Cl + Cr) * L1 * L2 * s**3 + (C2 * (C1 + Cl) + (C1 + C2 + Cl) * Cr) * L1 * L2 * R2 * s**4))

    def tf2(self, f):
        s = 1j * 2. * np.pi * f
        Cl = self.Cl
        Cr = self.Cr
        R1 = self.R1
        L1 = self.L1
        C1 = self.C1
        R2 = self.R2
        L2 = self.L2
        C2 = self.C2
        return (Cl * Cr * L1 * L2 * R1 * R2 * s**4) / (L1 * s * (R2 + L2 * s + (C2 + Cr) * L2 * R2 * s**2) + R1 * (R2 + L2 * s + ((C1 + Cl + Cr) * L1 + (C2 + Cr) * L2) * R2 * s**2 + (C1 + Cl + Cr) * L1 * L2 * s**3 + (C2 * (C1 + Cl) + (C1 + C2 + Cl) * Cr) * L1 * L2 * R2 * s**4))


# Load the C library and set arguments and return types
curr_folder = os.path.realpath(os.path.dirname(__file__))
lib_path = os.path.join(curr_folder, "sim_cvode.so")
c_lib = ctypes.cdll.LoadLibrary(lib_path)
c_lib.integrate_cvode.restype = ctypes.c_int
c_lib.integrate_cvode.argtypes = [
    ctypes.c_void_p,  # para
    c_double,  # T0
    npct.ndpointer(c_double, ndim=1, flags="C_CONTIGUOUS"),  # y0
    c_double,  # reltol
    c_double,  # abstol
    ctypes.c_int,  # nout
    c_double,  # dt
    npct.ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),  # outdata
]


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
