from __future__ import division, print_function

from copy import copy
import ctypes
import os
import sys
import time

import numpy as np
import numpy.ctypeslib as npct
from scipy.constants import Boltzmann

ID_NOT_SET = -1

NEQ = 5

ID_LINEAR = 0
ID_DUFFING = 1
ID_JOSEPHSON = 2
ID_JOSEPHSON_BOTH = 3

ID_NO_DRIVE = 0
ID_LOCKIN = 1
ID_DRIVE_V = 2

c_double = ctypes.c_double
c_double_p = ctypes.POINTER(ctypes.c_double)
c_double_pp = ctypes.POINTER(c_double_p)

PHI0 = 2.067833831e-15  # Wb, magnetic flux quantum
T0 = 1e-9  # s, time scale
F0 = 1e9  # Hz, frequency scale


class SimulationParameters(object):
    """ Container class for the simulation parameters and to run the
    simulation.

    Attributes:
        para (_SimPara): the C structure to be passed to the cvode simulator.
    """

    def __init__(
        self,
        Cl, Cr,
        R1, L1, C1,
        R2, L2, C2,
        R0=50., fs=50e9,
    ):
        """
        Args:
            Cl (float): left coupling capacitance (drive to V_1) in farad (F)
            Cr (float): right coupling capacitance (V_1 to V_2) in farad (F)
            R1 (float): cavity resistance in ohm (Omega)
            L1 (float): cavity inductance in henry (H)
            C1 (float): cavity capacitance in farad (F)
            R2 (float): qubit resistance in ohm (Omega)
            L2 (float): qubit inductance in henry (H)
            C2 (float): qubit capacitance in farad (F)
            R0 (float, optional): transmission line impedance in ohm (Omega)
            fs (float, optional): sampling frequency in hertz (Hz)
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
        self.R0 = R0

        self.w01_b = np.sqrt(1. / (self.C1 * self.L1))  # rad/s, bare cavity 1 resonance frequency
        self.f01_b = self.w01_b / 2. / np.pi  # Hz, bare cavity 1 resonance frequency
        self.Q1_b = self.R1 * np.sqrt(self.C1 / self.L1)  # bare cavity 1 quality factor
        self.w02_b = np.sqrt(1. / (self.C2 * self.L2))  # rad/s, bare cavity 2 resonance frequency
        self.f02_b = self.w02_b / 2. / np.pi  # Hz, bare cavity 2 resonance frequency
        self.Q2_b = self.R2 * np.sqrt(self.C2 / self.L2)  # bare cavity 2 quality factor

        self.w01_d = np.sqrt(1. / (self.Csum1 * self.L1))  # rad/s, dressed cavity 1 resonance frequency
        self.f01_d = self.w01_d / 2. / np.pi  # Hz, dressed cavity 1 resonance frequency
        self.Q1_d = self.R1 * np.sqrt(self.Csum1 / self.L1)  # dressed cavity 1 quality factor
        self.w02_d = np.sqrt(1. / (self.Csum2 * self.L2))  # rad/s, dressed cavity 2 resonance frequency
        self.f02_d = self.w02_d / 2. / np.pi  # Hz, dressed cavity 2 resonance frequency
        self.Q2_d = self.R2 * np.sqrt(self.Csum2 / self.L2)  # dressed cavity 2 quality factor

        self.Nbeats = 1  # nr of windows (periods, beats) to simulate

        self.fs = float(fs)  # Hz
        self.df = 50e6  # Hz
        self.dw = 2. * np.pi * self.df
        self.df = self.dw / (2. * np.pi)
        self.ns = int(round(self.fs / self.df))  # nr of samples in one windows (period, beat)
        self.dt = 1. / self.fs
        self.T = 1. / self.df
        self.ws = 2. * np.pi * self.fs

        self.drive_id = ID_NOT_SET

        self.nr_drives = 0
        self.w_arr = None
        self.A_arr = None
        self.P_arr = None

        self.drive_V_arr = None

        self.sys_id = ID_LINEAR
        self.duff1 = 0.
        self.duff2 = 0.
        self.phi0 = 1.

        self.add_thermal_noise = False
        self.noise_T1 = 0.  # K
        self.noise_T2 = 0.  # K
        self.noise_T0 = 0.  # K
        self.noise1_array = None
        self.noise2_array = None
        self.noise0_array = None

        self.next_init = np.zeros(NEQ)

        # Populate C structure
        self.para = _SimPara()

        self.para.stiff_equation = True

        self.para.Cl = float(self.Cl)
        self.para.Cr = float(self.Cr)
        self.para.R1 = float(self.R1)
        self.para.L1 = float(self.L1)
        self.para.C1 = float(self.C1)
        self.para.R2 = float(self.R2)
        self.para.L2 = float(self.L2)
        self.para.C2 = float(self.C2)
        self.para.R0 = float(self.R0)

        self.para.drive_id = int(self.drive_id)

        self.para.nr_drives = int(self.nr_drives)
        self.para.w_arr = c_double_p()
        self.para.A_arr = c_double_p()
        self.para.P_arr = c_double_p()

        self.para.drive_V_arr = c_double_p()

        self.para.sys_id = int(self.sys_id)

        self.para.duff1 = float(self.duff1)
        self.para.duff2 = float(self.duff2)
        self.para.phi0 = float(self.phi0)

        self.para.add_thermal_noise = self.add_thermal_noise
        self.para.noise1_array = c_double_p()
        self.para.noise2_array = c_double_p()
        self.para.noise0_array = c_double_p()

    def pickable_copy(self):
        """ Return a pickable copy of this object. Useful e.g. to save the
        simulation parameters with numpy.savez or with pickle.dump

        Returns:
            (SimulationParameters)

        Notes:
            The returned object cannot be used to run simulations.
            Bad things will happen, probably...
        """
        new = copy(self)
        new.para = None
        return new

    def get_time_arr(self, extra=False):
        """ Return the simulated time array.

        Args:
            extra (bool, optional): add an extra time sample at the end

        Returns:
            (np.ndarray): simulation-time array in seconds (s), with length
                Nbeats * ns (or Nbeats * ns + 1 if extra=True)

        See also:
            get_drive_time_arr

        Examples:
            >>> para = SimulationParameters(
                    Cl=1e-15, Cr=1e-12,
                    R1=3162., L1=1e-9, C1=1e-12,
                    R2=3162., L2=2e-9, C2=2e-12,
                    w_arr=[1e9,], A_arr=[0.0005,], P_arr=[0.,],
                )
            >>> sol = para.simulate()
            >>> t_sol = para.get_time_arr()
            >>> fig, ax = plt.subplots()
            >>> ax.plot(t_sol, sol[:, 1])
            >>> fig.show()
        """
        if extra:
            return self.dt * np.arange(self.Nbeats * self.ns + 1)
        else:
            return self.dt * np.arange(self.Nbeats * self.ns)

    def get_drive_time_arr(self):
        """ Return the time array needed to compute the drive signal.

        Returns:
            (np.ndarray): drive-time array in seconds (s), with length
                Nbeats * ns + 1

        Notes:
            equivalent to get_time_arr(extra=True)

        Examples:
            >>> para = SimulationParameters(
                    Cl=1e-15, Cr=1e-12,
                    R1=3162., L1=1e-9, C1=1e-12,
                    R2=3162., L2=2e-9, C2=2e-12,
                    w_arr=[1e9,], A_arr=[0.0005,], P_arr=[0.,],
                )
            >>> t_drive = para.get_drive_time_arr()
            >>> drive = 0.0005 * np.cos(1e9 * t_drive)
            >>> para.set_drive_V(drive)
            >>> sol = para.simulate()
            >>> t_sol = para.get_time_arr()
            >>> fig, ax = plt.subplots()
            >>> ax.plot(t_sol, sol[:, 1])
            >>> fig.show()
        """
        return self.get_time_arr(extra=True)

    def set_Nbeats(self, Nbeats):
        """ Set the number of measurement windows (or beats, or periods) to
        simulate.

        Args:
            Nbeats (int)
        """
        self.Nbeats = int(Nbeats)

    def set_df(self, df):
        """ Set measurement bandwidth (frequency resolution) in hertz.

        Args:
            df (float): Hz
        """
        self.set_dw(2. * np.pi * df)

    def set_dw(self, dw):
        """ Set measurement bandwidth (frequency resolution) in radians per
        second.

        Args:
            dw (float): rad/s
        """
        self.dw = float(dw)
        self.df = self.dw / (2. * np.pi)
        self.ns = int(round(self.fs / self.df))
        self.T = 1. / self.df

    def _set_drive_frequencies(self, f_arr):
        f_arr = np.atleast_1d(f_arr)
        self._set_w_arr(2. * np.pi * f_arr)

    def _set_w_arr(self, w_arr):
        w_arr = np.atleast_1d(w_arr).copy()
        assert len(w_arr.shape) == 1
        assert len(w_arr) == self.nr_drives
        self.w_arr = w_arr
        self.para.w_arr = npct.as_ctypes(self.w_arr)

    def set_drive_none(self):
        """ Set drive to zero. Useful e.g. for simulating only noise.
        """
        self.drive_id = ID_NO_DRIVE
        self.para.drive_id = int(self.drive_id)

    def set_drive_lockin(self, f_arr, A_arr, P_arr):
        """ Set a lockin drive, calculated in real time during the simulation.

        Args:
            f_arr (array_like): drive frequencies in hertz (Hz)
            A_arr (array_like): drive amplitudes in volts (V)
            P_arr (array_like): drive phases in radians (rad)

        Notes:
            No tuning is enforced!
            Make sure to tune the frequencies with the method tune.

        Raises:
            ValueError: if the arrays have different shape

        See also:
            tune
            set_drive_V
        """
        w_arr = 2. * np.pi * np.atleast_1d(f_arr).astype(np.float64)
        if len(w_arr.shape) > 1:
            raise ValueError("Arrays must be 1D.")
        A_arr = np.atleast_1d(A_arr).astype(np.float64)
        if A_arr.shape != w_arr.shape:
            raise ValueError("Arrays must have same shape.")
        P_arr = np.atleast_1d(P_arr).copy()
        if P_arr.shape != w_arr.shape:
            raise ValueError("Arrays must have same shape.")

        self.nr_drives = len(w_arr)
        self.w_arr = w_arr
        self.A_arr = A_arr
        self.P_arr = P_arr
        self.drive_id = ID_LOCKIN

        self.para.drive_id = int(self.drive_id)
        self.para.nr_drives = self.nr_drives
        self.para.w_arr = npct.as_ctypes(self.w_arr)
        self.para.A_arr = npct.as_ctypes(self.A_arr)
        self.para.P_arr = npct.as_ctypes(self.P_arr)

    def set_drive_V(self, V):
        """ Set the drive voltage. Its derivative is calculated in real time
        during the simulation from a cubic spline interpolation.

        Args:
            V (array_like): drive signal in volts (V), must be evaluated at
                times given by get_drive_time_arr

        Notes:
            One extra sample is needed to obtain the next initial condition.
            Use get_drive_time_arr to obtain the times at which the drive
            signal must be defined.

        Raises:
            ValueError: if the array has wrong shape

        See also:
            get_drive_time_arr
            set_drive_lockin
        """
        V = np.atleast_1d(V).astype(np.float64)
        if len(V.shape) > 1:
            raise ValueError("Array must be 1D.")
        if len(V) != self.Nbeats * self.ns + 1:
            raise ValueError("Array must have same shape as get_drive_time_arr().")

        self.drive_V_arr = V
        self.drive_id = ID_DRIVE_V
        self.para.drive_V_arr = npct.as_ctypes(self.drive_V_arr)
        self.para.drive_id = int(self.drive_id)

    def get_drive_V(self):
        """ Get the drive voltage used in the simulation.

        Returns:
            (np.ndarray): with shape (Nbeats * ns + 1, )

        Notes:
            One extra sample is needed to obtain the next initial condition.
            Use get_drive_time_arr to obtain the times at which the drive
            signal must be defined.
        """
        if self.drive_id == ID_DRIVE_V:
            return self.drive_V_arr
        else:  # ID_LOCKIN
            t = self.get_drive_time_arr()
            Vg = np.zeros_like(t)
            for ii in range(self.nr_drives):
                Vg += self.A_arr[ii] * np.cos(self.w_arr[ii] * t + self.P_arr[ii])
            return Vg

    def set_linear(self):
        """ Set the second oscillator (qubit) as linear.

        Notes:
            This is the default setting.
            The current through the linear inductor is:
            I_L2 = Phi_2 / L2
        """
        self.sys_id = ID_LINEAR
        self.para.sys_id = int(self.sys_id)

    def set_duffing(self, duff, duff2=None):
        """ Set a Duffing nonlinearity on the inductors.

        Args:
            duff (float): cavity Duffing coefficient in Wb**(-2)
            duff2 (float, optional): qbit Duffing coefficient in Wb**(-2),
                If None, use the same as the cavity

        Notes:
            The current through the nonlinear inductor is:
            I_L2 = Phi_2 / L2 * (1. - duff2 * Phi_2**2)
        """
        self.duff1 = float(duff)
        if duff2 is None:
            duff2 = duff
        self.duff2 = float(duff2)
        self.para.duff1 = float(self.duff1)
        self.para.duff2 = float(self.duff2)
        self.sys_id = ID_DUFFING
        self.para.sys_id = int(self.sys_id)

    def set_josephson(self, PHI0=2.067833831e-15, which='right'):
        """ Set a Josephson-junction nonlinearity on the second oscillator
        (qubit).

        Args:
            PHI0 (float, optional): magnetic flux quantum,
                default to 2.067833831e-15 Wb
            which (str, optional): set the nonlinearity in the 'right'
                (second, qubit) or 'both' oscillators

        Notes:
            The current through the nonlinear inductor is:
            I_L2 = I_C * sin(2. * np.pi * Phi_2 / PHI0)
                 = PHI0 / (2. * np.pi * L_2) * sin(2. * np.pi * Phi_2 / PHI0)
        """
        if which not in ['right', 'both']:
            raise ValueError("only 'right' and 'both' are allowed.")
        self.phi0 = float(PHI0)
        self.para.phi0 = float(self.phi0)
        if which == 'right':
            self.sys_id = ID_JOSEPHSON
        else:  # 'both'
            self.sys_id = ID_JOSEPHSON_BOTH
        self.para.sys_id = int(self.sys_id)

    def set_noise_T(self, T, T2=None, T0=None, seed=None):
        """ Set strength of the thermal (process) noise on the two oscillators
        (cavity and qubit);

        Args:
            T (float): cavity noise temperature in kelvin (K). Set to zero to
                turn of the simulation of the noise (default).
            T2 (float, optional): qubit noise temperature in kelvin (K).
                If None, use the same as the cavity
            T0 (float, optional): drive noise temperature in kelvin (K).
                If None, use the same as the cavity
            seed (int, optional): seed the random-number generator in NumPy

        Notes:
            The noise is implemented as a voltage source in series with the
            resistor in the oscillator (Thevenin equivalent).
            The noise voltage V_N is a sample from a Gaussian distribution
            with mean zero and variance
            sigma**2 / 2. * f_S
            where f_S is the sampling frequency in Hz and sigma**2 is the
            one-sided power spectral density of the noise in V**2 / Hz
            sigma**2 = 4. * k_B * T * R
            where k_B = 1.38064852e-23 J/K is the Boltzmann constant.
            R is R1, R2 or R0 for cavity, qbit or drive, respectively.
        """
        if T2 is None:
            T2 = T
        if T0 is None:
            T0 = T
        self.noise_T1 = float(T)
        self.noise_T2 = float(T2)
        self.noise_T0 = float(T0)

        if self.noise_T1 or self.noise_T2 or self.noise_T0:
            PSDv1_onesided = 4. * Boltzmann * self.noise_T1 * self.R1
            PSDv2_onesided = 4. * Boltzmann * self.noise_T2 * self.R2
            PSDv0_onesided = 4. * Boltzmann * self.noise_T0 * self.R0
            PSDv1_twosided = PSDv1_onesided / 2.
            PSDv2_twosided = PSDv2_onesided / 2.
            PSDv0_twosided = PSDv0_onesided / 2.
            self.add_thermal_noise = True
            self.para.add_thermal_noise = True
            np.random.seed(seed)
            self.noise1_array = np.sqrt(PSDv1_twosided) * np.sqrt(self.fs) * np.random.randn(self.Nbeats * self.ns + 2)
            self.noise2_array = np.sqrt(PSDv2_twosided) * np.sqrt(self.fs) * np.random.randn(self.Nbeats * self.ns + 2)
            self.noise0_array = np.sqrt(PSDv0_twosided) * np.sqrt(self.fs) * np.random.randn(self.Nbeats * self.ns + 2)
            self.para.noise1_array = npct.as_ctypes(self.noise1_array)
            self.para.noise2_array = npct.as_ctypes(self.noise2_array)
            self.para.noise0_array = npct.as_ctypes(self.noise0_array)
        else:
            self.add_thermal_noise = False
            self.para.add_thermal_noise = False
            self.noise1_array = None
            self.noise2_array = None
            self.noise0_array = None
            self.para.noise1_array = c_double_p()
            self.para.noise2_array = c_double_p()
            self.para.noise0_array = c_double_p()

    def simulate(self,
                 init=None, continue_run=False,
                 rtol=1.49012e-8, atol=1.49012e-8,
                 rescale=True, print_time=False):
        """ Run the simulation. Can be time consuming!

        Args:
            init (np.ndarray, optional): initial condition for the integrator,
                if None use np.zeros(NEQ) unless continue_run=True.
                OBS: the data will be overwritten with the initial condition
                for the next run!
            continue_run (bool, optional): use the end of the last run as
                initial condition. Ignored if init is provided.
            rtol (float, optional): scalar relative tolerance.
            atol (float, optional): scalar absolute tolerance.
            rescale (bool, optional): run simulation in scaled units. Set to
                False to run in SI units (not recommended). See notes below.
            print_time (bool, optional): print the computation time to stdout.

        Notes:
            For info on setting tolerances, see
            https://computation.llnl.gov/projects/sundials/faq#cvode_tols
            When rescale=True, the simulation is run in scaled units:
            - time in units of T0 = 1 ns,
            - flux in units of PHI0 = 2.067833831e-15 Wb.
            As a consequence, other quantities are:
            - frequency in units of F0 = 1 / T0 = 1 GHz,
            - voltage in units of PHI0 * F0,
            - capacitance in nanofarad (nF),
            - inductance in nanohenry (nH),
            - duffing term in units of PHI0**(-2).
            All the relevant quantities are rescaled before the simulation,
            and scaled back to SI units after the calculation. Setting
            rescale=False turns off the rescaling and is not recommended,
            however it can be useful for debugging purposes.

        Returns:
            (np.ndarray): the solution array, with shape (Nbeats * ns, NEQ).
        """
        if self.drive_id == ID_NOT_SET:
            raise RuntimeError("No drive initialized! Use one of set_drive_none, set_drive_lockin, set_drive_V, set_drive_dVdt.")
        if self.add_thermal_noise:
            if len(self.noise1_array) != self.Nbeats * self.ns + 2:
                raise RuntimeError("Wrong noise-array length! Regenerate or disable noise with set_noise_T.")
        if self.drive_id == ID_DRIVE_V:
            if len(self.drive_V_arr) != self.Nbeats * self.ns + 1:
                raise RuntimeError("Wrong drive-array length! Reset drive with set_drive_V or set_drive_dVdt.")
        if init is None:
            if continue_run:
                init = self.next_init
            else:
                init = np.zeros(NEQ)
        else:
            if not isinstance(init, np.ndarray):
                raise TypeError("init must be a NumPy array!")
            if not init.dtype == np.float64:
                raise TypeError("init must be of type np.float64!")
            if not init.shape == (NEQ,):
                raise TypeError("init must have shape (NEQ,)!")
        t = self.dt * np.arange(self.Nbeats * self.ns + 1)
        out = np.empty((len(t), NEQ))

        assert init.shape[0] == NEQ
        assert t.shape[0] == out.shape[0]
        assert out.shape[1] == NEQ

        # rescale to simulation units
        if rescale:
            t /= T0
            init[0] /= PHI0 * F0
            init[1] /= PHI0
            init[2] /= PHI0 * F0
            init[3] /= PHI0
            init[4] /= PHI0 * F0
            self.para.Cl /= T0
            self.para.Cr /= T0
            self.para.L1 /= T0
            self.para.C1 /= T0
            self.para.L2 /= T0
            self.para.C2 /= T0
            if self.drive_id == ID_LOCKIN:
                self.w_arr /= F0
                self.A_arr /= PHI0 * F0
            elif self.drive_id == ID_DRIVE_V:
                self.drive_V_arr /= PHI0 * F0
            self.para.duff1 *= PHI0**2
            self.para.duff2 *= PHI0**2
            self.para.phi0 /= PHI0
            if self.add_thermal_noise:
                self.noise1_array /= PHI0 * F0
                self.noise2_array /= PHI0 * F0
                self.noise0_array /= PHI0 * F0

        t0 = time.time()
        res = c_lib.integrate_cvode(
            ctypes.byref(self.para),
            init, t,
            out, len(t),
            float(rtol), float(atol),
        )
        t1 = time.time()
        if print_time:
            dt = t1 - t0
            print(format_sec(dt))
        if res:
            print("*** Some error occurred in CVODE. Return flag: {:d}".format(res))

        init[:] = out[-1, :]

        # rescale back to SI units
        if rescale:
            t *= T0
            init[0] *= PHI0 * F0
            init[1] *= PHI0
            init[2] *= PHI0 * F0
            init[3] *= PHI0
            init[4] *= PHI0 * F0
            out[:, 0] *= PHI0 * F0
            out[:, 1] *= PHI0
            out[:, 2] *= PHI0 * F0
            out[:, 3] *= PHI0
            out[:, 4] *= PHI0 * F0
            self.para.Cl *= T0
            self.para.Cr *= T0
            self.para.L1 *= T0
            self.para.C1 *= T0
            self.para.L2 *= T0
            self.para.C2 *= T0
            if self.drive_id == ID_LOCKIN:
                self.w_arr *= F0
                self.A_arr *= PHI0 * F0
            elif self.drive_id == ID_DRIVE_V:
                self.drive_V_arr *= PHI0 * F0
            self.para.duff1 /= PHI0**2
            self.para.duff2 /= PHI0**2
            self.para.phi0 *= PHI0
            if self.add_thermal_noise:
                self.noise1_array *= PHI0 * F0
                self.noise2_array *= PHI0 * F0
                self.noise0_array *= PHI0 * F0

        self.next_init = init.copy()
        return out[:-1, :]

    def tune(self, f, df, priority='df', rounding=1, regular=False):
        """Performs frequency tuning.

        Args:
            f (float): target frequency in hertz (Hz).
            df (float): target measurement bandwidth in hertz (Hz).
            priority (str, optional): tuning priority, see Usage.
            rounding (int, optional): if set to m > 1, it will ensure Fourier
                leakage is minimized also in case of using a downsampling of m
            regular (bool, optional): require that the number of samples per
                pixel is a regular number, see Notes.

        Returns:
            f_tuned (float): The tuned frequency in hertz (Hz).
            df_tuned (float): The tuned measurment bandwidth in hertz (Hz).

        Usage:
            The returned bandwidth and frequencies are tuned so that they
            fulfill the condition f = n * df which avoids Fourier leakage.
            With priority 'f' the highest priority is put on getting the
            frequency as close as possible to the target frequency.
            With priority 'df' the highest priority is put on getting the
            measurement bandwidth as close as possible to the target value.

        Notes:
            If simulation result is used in an FFT calculation, setting
            regular=True ensures that the number of samples per pixel ns has
            only 2, 3 or 5 as prime factors (ns is 5-smooth, or a regular
            number). This can significantly speed up the FFT calculation.

        Examples:
            >>> f_out, df_out = para.tune(1e9, 1e6)
            >>> para.set_df(df_out)
            >>> para.set_drive_lockin([f_out,], [0.1,], [0.])
        """
        if priority not in ['df', 'f']:
            raise ValueError("Allowed priorities are 'df' and 'f'.")
        f = float(f)
        df = float(df)
        rounding = int(rounding)

        if f and priority == 'f':
            n = int(round(f / df))
            df = f / n
            ns = int(round(self.fs / df / rounding)) * rounding
            if regular:
                ns = _closest_regular(ns, rounding=rounding)
            df_out = self.fs / ns
            n = int(round(f / df_out))
            f_out = n * df_out
        else:
            ns = int(round(self.fs / df / rounding)) * rounding
            if regular:
                ns = _closest_regular(ns, rounding=rounding)
            df_out = self.fs / ns
            n = int(round(f / df_out))
            f_out = n * df_out
        return f_out, df_out

    def tf0(self, f):
        """ Linear response function from the drive voltage V_G to the voltage
        on the load V_0.

        Args:
            f (float or np.ndarray): frequency in hertz (Hz)

        Returns:
            (float or np.ndarray)
        """
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n0 = r1 * r2
        n1 = (l2 * r1 + l1 * r2)
        n2 = (l1 * l2 + ((c1 + cL + cR) * l1 + (c2 + cR) * l2) * r1 * r2)
        n3 = ((c1 + cL + cR) * l1 * l2 * r1 + (c2 + cR) * l1 * l2 * r2)
        n4 = (c2 * (c1 + cL) + (c1 + c2 + cL) * cR) * l1 * l2 * r1 * r2

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n0 + n1 * s + n2 * s**2 + n3 * s**3 + n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tf1(self, f):
        """ Linear response function from the drive voltage V_G to the voltage
        on the first oscillator (cavity) V_1.

        Args:
            f (float or np.ndarray): frequency in hertz (Hz)

        Returns:
            (float or np.ndarray)
        """
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n2 = cL * l1 * r1 * r2
        n3 = cL * l1 * l2 * r1
        n4 = cL * (c2 + cR) * l1 * l2 * r1 * r2

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n2 * s**2 + n3 * s**3 + n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tf2(self, f):
        """ Linear response function from the drive voltage V_G to the voltage
        on the second oscillator (qubit) V_2.

        Args:
            f (float or np.ndarray): frequency in hertz (Hz)

        Returns:
            (float or np.ndarray)
        """
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n4 = cL * cR * l1 * l2 * r1 * r2

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tfr(self, f):
        """ Linear response function from the drive voltage V_G to the voltage
        reflected from the load V_0^-.

        Args:
            f (float or np.ndarray): frequency in hertz (Hz)

        Returns:
            (float or np.ndarray)
        """
        return self.tf0(f) - 0.5

    def tfn00(self, f):
        return self.tf0(f)

    def tfn01(self, f):
        return self.tf1(f)

    def tfn02(self, f):
        return self.tf2(f)

    def tfn10(self, f):
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n2 = cL * l1 * r0 * r2
        n3 = cL * l1 * l2 * r0
        n4 = cL * (c2 + cR) * l1 * l2 * r0 * r2

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n2 * s**2 + n3 * s**3 + n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tfn11(self, f):
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n1 = l1 * r2
        n2 = l1 * (l2 + cL * r0 * r2)
        n3 = l1 * (cL * l2 * r0 + (c2 + cR) * l2 * r2)
        n4 = cL * (c2 + cR) * l1 * l2 * r0 * r2

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n1 * s + n2 * s**2 + n3 * s**3 + n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tfn12(self, f):
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n3 = cR * l1 * l2 * r2
        n4 = cL * cR * l1 * l2 * r0 * r2

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n3 * s**3 + n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tfn20(self, f):
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n4 = cL * cR * l1 * l2 * r0 * r1

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tfn21(self, f):
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n3 = cR * l1 * l2 * r1
        n4 = cL * cR * l1 * l2 * r0 * r1

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n3 * s**3 + n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)

    def tfn22(self, f):
        s = 1j * 2. * np.pi * f
        r0 = self.R0
        r1 = self.R1
        r2 = self.R2
        c1 = self.C1
        c2 = self.C2
        cL = self.Cl
        cR = self.Cr
        l1 = self.L1
        l2 = self.L2

        n1 = l2 * r1
        n2 = l2 * (l1 + cL * r0 * r1)
        n3 = l1 * l2 * (cL * r0 + (c1 + cL + cR) * r1)
        n4 = cL * (c1 + cR) * l1 * l2 * r0 * r1

        d0 = r1 * r2
        d1 = (l1 * r2 + r1 * (l2 + cL * r0 * r2))
        d2 = (l1 * l2 + cL * l1 * r0 * r2 + r1 * (cL * l2 * r0 + (c1 + cL + cR) * l1 * r2 + (c2 + cR) * l2 * r2))
        d3 = (cL * l1 * l2 * r0 + (c2 + cR) * l1 * l2 * r2 + r1 * ((c1 + cL + cR) * l1 * l2 + cL * ((c1 + cR) * l1 + (c2 + cR) * l2) * r0 * r2))
        d4 = (cL * (c2 + cR) * l1 * l2 * r0 * r2 + l1 * l2 * r1 * (cL * (c1 + cR) * r0 + c2 * (c1 + cL) * r2 + (c1 + c2 + cL) * cR * r2))
        d5 = cL * (c2 * cR + c1 * (c2 + cR)) * l1 * l2 * r0 * r1 * r2

        return (n1 * s + n2 * s**2 + n3 * s**3 + n4 * s**4) / (d0 + d1 * s + d2 * s**2 + d3 * s**3 + d4 * s**4 + d5 * s**5)


# Load the C library and set arguments and return types
curr_folder = os.path.realpath(os.path.dirname(__file__))
if sys.platform == 'win32':
    my_ext = '.dll'
    lib_path = os.path.join(curr_folder, "win64", "lib")
    if lib_path not in os.environ['path']:
        os.environ['path'] = ";".join((os.environ['path'], lib_path))
elif sys.platform == 'darwin':
    my_ext = '.bundle'
elif sys.platform.startswith('linux'):
    my_ext = '.so'
load_path = os.path.join(curr_folder, "sim_cvode" + my_ext)
c_lib = ctypes.cdll.LoadLibrary(load_path)
c_lib.integrate_cvode.restype = ctypes.c_int
c_lib.integrate_cvode.argtypes = [
    ctypes.c_void_p,  # para
    npct.ndpointer(c_double, ndim=1, flags="C_CONTIGUOUS"),  # y0
    npct.ndpointer(c_double, ndim=1, flags="C_CONTIGUOUS"),  # tout_arr
    npct.ndpointer(c_double, ndim=2, flags="C_CONTIGUOUS"),  # outdata
    ctypes.c_int,  # nout
    c_double,  # reltol
    c_double,  # abstol
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


class _SimPara(ctypes.Structure):
    """ Structure used to pass the relevant simulation parameters
    to the C code. The fields here MUST match the fields of the
    SimPara structure in the C code: same name, type and ORDER!
    """
    _fields_ = [
        # Solver
        ('stiff_equation', ctypes.c_int),  # bool

        # Circuit
        ('Cl', c_double),  # F
        ('Cr', c_double),  # F
        ('R1', c_double),  # ohm
        ('L1', c_double),  # H
        ('C1', c_double),  # F
        ('R2', c_double),  # ohm
        ('L2', c_double),  # H
        ('C2', c_double),  # F
        ('R0', c_double),  # ohm

        # Drive
        ('drive_id', ctypes.c_int),  # ID_LOCKIN, ID_DRIVE_V
        # Lockin
        ('nr_drives', ctypes.c_int),
        ('w_arr', c_double_p),  # rad/s
        ('A_arr', c_double_p),  # V
        ('P_arr', c_double_p),  # rad
        # Drive_V or Drive_dVdt
        ('drive_V_arr', c_double_p),  # V or V/s
        ('drive_spline', ctypes.c_void_p),  # internal
        ('drive_acc', ctypes.c_void_p),  # internal

        # Select system
        ('sys_id', ctypes.c_int),  # ID_LINEAR, ID_DUFFING, ID_JOSEPHSON or ID_JOSEPHSON_BOTH

        # Duffing: V/L * (1. - duff * V*V)
        ('duff1', c_double),  # V^-2
        ('duff2', c_double),  # V^-2

        # Josephson
        ('phi0', c_double),

        # Thermal noise
        ('add_thermal_noise', ctypes.c_int),  # bool
        ('noise1_array', c_double_p),  # V
        ('noise1_spline', ctypes.c_void_p),  # internal
        ('noise1_acc', ctypes.c_void_p),  # internal
        ('noise2_array', c_double_p),  # V
        ('noise2_spline', ctypes.c_void_p),  # internal
        ('noise2_acc', ctypes.c_void_p),  # internal
        ('noise0_array', c_double_p),  # V
        ('noise0_spline', ctypes.c_void_p),  # internal
        ('noise0_acc', ctypes.c_void_p),  # internal

        # Other internal
        ('b', c_double * NEQ),
        ('a', c_double * NEQ * NEQ),
    ]


def _is_regular(n):
    """ Test whether n is a regular number.

    Args:
        n (int)

    Returns:
        (bool)

    Regular numbers are also known as 5-smooth numbers or Hamming numbers.
    They are the numbers whose only prime divisors are 2, 3, and 5.
    The FFT of arrays whose length is a regular number can be calculated
    efficiently, see https://doi.org/10.1137/0913039 and
    http://www.fftw.org/fftw3_doc/Complex-DFTs.html#Complex-DFTs

    See also:
        closest_regular
        next_regular
        previous_regular
    """
    while not(n % 2):
        n //= 2
    while not(n % 3):
        n //= 3
    while not(n % 5):
        n //= 5
    return n == 1


def _next_regular(n, rounding=1):
    """ Find the smallest regular number m such that m >= n.

    Args:
        n (int)
        rounding (int, optional): require that m is divisible by rounding.

    Returns:
        m (int)

    See also:
        is_regular
        closest_regular
        previous_regular
    """
    if not _is_regular(rounding):
        raise ValueError('rounding must be a regular number.')
    while (n % rounding or not _is_regular(n)):
        n += 1
    return n


def _previous_regular(n, rounding=1):
    """ Find the largest regular number m such that m <= n.

    Args:
        n (int)
        rounding (int, optional): require that m is divisible by rounding.

    Returns:
        m (int)

    See also:
        is_regular
        closest_regular
        next_regular
    """
    if not _is_regular(rounding):
        raise ValueError('rounding must be a regular number.')
    if n < rounding:
        raise ValueError('n must be >= rounding')
    if n < 1:
        raise ValueError('n must be >= 1')
    while (n % rounding or not _is_regular(n)):
        n -= 1
    return n


def _closest_regular(n, rounding=1):
    """ Find the regular number m such that abs(m-n) is smallest.

    Args:
        n (int)
        rounding (int, optional): require that m is divisible by rounding.

    Returns:
        m (int)

    See also:
        is_regular
        next_regular
        previous_regular
    """
    if not _is_regular(rounding):
        raise ValueError('rounding must be a regular number.')
    next_r = _next_regular(n, rounding=rounding)
    try:
        prev_r = _previous_regular(n, rounding=rounding)
    except ValueError:
        return next_r
    if (next_r - n) < (n - prev_r):
        return next_r
    else:
        return prev_r
