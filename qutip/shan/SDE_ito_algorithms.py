# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:58:40 2018

@author: Shan
"""
import numpy as np
import qutip as qt


#function to numerically integrate an ito SDE using a Runge-Kutta algorithm
#see Wikipedia for details or ......
def rk_sde(H_function,
           H_drive_function,
           rho,
           dissipation_list,
           time_list,
           phi=0,
           detection_efficiency=1,
           homodyne=True):
    """
    Numerically integrates the quantum stochastic master equation with homodyne
    detection using the Ito Runge-Kutta algorithm.
    See Wikipedia or ..... for details
    hbar=1, working in units of frequency

    Parameters
    ----------
    H_function : time-independent operator function
        Hamiltonian governing dissipationless evolution. Should be a function.

    H_drive_function : time-dependent operator function
        Hamiltonian contribution due to the presence of an external drive.
        Should be a function of the variable in time_list. Evaluated at
        time_list[0] should be the initial drive Hamiltonian.
    rho : operator
        Density matrix of quantum state at time_list[0]
    dissipation_list : array of operators
        An array or list of dissipation operators with factor np.sqrt('rate').
        As of now, the FIRST ELEMENT IS THE SIGNAL WE ARE HOMODYNING!!
        I.e. dissipation_list[0]
    time_list : scalar, array
        Times at which the stochastic differential equation and hamiltonian
        should be evaluated. NOT compatible with variable time step.
        Time step is thus defined as the difference between each sample.
    phi : scalar
        A number specifying the local oscillator phase. If phi=0, we are
        conditioning the state on the measured I-quadrature.
    detection_efficiency : scalar
        A number <= 1 specifying the detection efficiency. If equals 1, the
        detector is 100% efficient
    homodyne : scalar
        1 indicates that a homodyne detection is desired, any other scalar
        will implement a heterodyne detection scheme

    Returns
    -------
    rho_t : operator list
        A list of density matrices at each time given in time_list.
    wiener_incr : float array
        An array of gaussian distributed random numbers with variance equal
        to the time step and mean equal to 0. Used to reconstruct the output
        signal used to condition the quantum state.
    """

    #state must be a density matrix, should include a check and raise error if not of the correct type
    #tyoe can be checked with rho.type

    #dissipation free part of the hamiltonian
    def H_time(t):
        return H_drive_function(t)  #+ H_function(t)

    #Liouville-von Neumann contribution
    def Liouville_von_Neumann(rho, hamiltonian):
        return -1j * qt.commutator(hamiltonian, rho, kind='normal')

    #dissipation on Lindblad form
    def lindblad_diss(rho, dissipator):
        return dissipator * rho * dissipator.dag() - 0.5 * qt.commutator(
            dissipator.dag() * dissipator, rho, kind='anti')

    #stochastic part / diffusive part of the Ito SDE
    #measurement induced dynamics enters here
    def stochastic_part(rho, operator, phi):
        LO_phase = np.exp(-1j * phi)
        sum1_part = LO_phase * operator * rho + rho * operator.dag(
        ) * LO_phase.conjugate()
        sum1_trace = qt.expect(rho, operator * LO_phase) + qt.expect(
            rho,
            operator.dag() * LO_phase.conjugate())
        return np.sqrt(detection_efficiency) * (sum1_part - sum1_trace * rho)

    #a small correction to avoid using derivatives of parameters (see algorithm for details)
    def Gamma_correction(rho, deterministic_part, stochastic_part, step_size):
        return rho + deterministic_part * step_size + stochastic_part * np.sqrt(
            step_size)

    #time step width
    width_t = np.abs(time_list[1] - time_list[0])

    #pre-generate a list of wiener increments
    #notice that the mean is zero
    #the standard deviation is sqrt(width_t), while variance is width_t
    #a list of gaussian distributed random numbers is generated
    if homodyne == 1:
        wiener_incr = np.random.normal(0, np.sqrt(width_t), (len(time_list)))
    else:
        wiener_x = np.random.normal(0, np.sqrt(width_t), (len(time_list)))
        wiener_y = np.random.normal(0, np.sqrt(width_t), (len(time_list)))

        wiener_incr = (wiener_x + 1j * wiener_y) / np.sqrt(2)

    #preparing a list for the density matrix
    #each entry will correspond to the density matrix at the corresponding time step
    rho_t = list(range(len(time_list)))
    rho_t[0] = rho

    #The hamiltonian evolves independently of the state and can therefore be computed now
    #time independent part:
    H_no_time = H_function(0)
    H_list = [H_time(time_entry) + H_no_time for time_entry in time_list]

    if homodyne == True:

        for index, time_entry in enumerate(time_list):

            if index < len(time_list) - 1:

                dissipation_contribution = sum([
                    lindblad_diss(rho_t[index], dissipator)
                    for dissipator in dissipation_list
                ])

                drift = Liouville_von_Neumann(
                    rho_t[index], H_list[index]) + dissipation_contribution

                diffusion = stochastic_part(rho_t[index], dissipation_list[0],
                                            phi)

                gamma = Gamma_correction(rho_t[index], drift, diffusion,
                                         width_t)

                stoch_gamma = stochastic_part(gamma, dissipation_list[0], phi)

                rho_t[index + 1] = rho_t[
                    index] + drift * width_t + diffusion * wiener_incr[index]
                +0.5 * (stoch_gamma - diffusion) * (
                    wiener_incr[index]**2 - width_t) / np.sqrt(width_t)

    else:

        for index, time_entry in enumerate(time_list):

            if index < len(time_list) - 1:

                dissipation_contribution = sum([
                    lindblad_diss(rho_t[index], dissipator)
                    for dissipator in dissipation_list
                ])

                drift = Liouville_von_Neumann(
                    rho_t[index], H_list[index]) + dissipation_contribution

                diffusion_x = stochastic_part(rho_t[index],
                                              dissipation_list[0], 0)

                diffusion_y = stochastic_part(rho_t[index],
                                              dissipation_list[0], np.pi / 2)

                gamma_x = Gamma_correction(rho_t[index], drift, diffusion_x,
                                           width_t)

                gamma_y = Gamma_correction(rho_t[index], drift, diffusion_y,
                                           width_t)

                stoch_gamma_x = stochastic_part(gamma_x, dissipation_list[0],
                                                0)

                stoch_gamma_y = stochastic_part(gamma_y, dissipation_list[0],
                                                np.pi / 2)

                rho_t[
                    index +
                    1] = rho_t[index] + drift * width_t + (1 / np.sqrt(2)) * (
                        diffusion_x * wiener_x[index] + 0.5 *
                        (stoch_gamma_x - diffusion_x) *
                        (wiener_x[index]**2 - width_t) / np.sqrt(width_t) +
                        diffusion_y * wiener_y[index] + 0.5 *
                        (stoch_gamma_y - diffusion_y) *
                        (wiener_y[index]**2 - width_t) / np.sqrt(width_t))

    return rho_t, wiener_incr


def rk_sde_qubit_trigger(H_function,
                         H_drive_function,
                         rho,
                         dissipation_list,
                         time_list,
                         N,
                         H_function_replace,
                         phi=0,
                         detection_efficiency=1):
    """
    As above, but forcibly causes the qubit to either decay or excite once state is determined.
    So it works only with the JC hamiltonian in the DISPERSIVE limit.
    Qubit system in first slot: (qubit, resonator)

    Parameters
    ----------
    H_function : time-independent operator function
        Hamiltonian governing dissipationless evolution. Should be a function.

    H_drive_function : time-dependent operator function
        Hamiltonian contribution due to the presence of an external drive.
        Should be a function of the variable in time_list. Evaluated at
        time_list[0] should be the initial drive Hamiltonian.
    rho : operator
        Density matrix of quantum state at time_list[0]
    dissipation_list : array of operators
        An array or list of dissipation operators with factor np.sqrt('rate').
        As of now, the FIRST ELEMENT IS THE SIGNAL WE ARE HOMODYNING!!
        I.e. dissipation_list[0]
    time_list : scalar, array
        Times at which the stochastic differential equation and hamiltonian
        should be evaluated. NOT compatible with variable time step.
        Time step is thus defined as the difference between each sample.
    N : Hilbert space dimension for resonator
    phi : scalar
        A number specifying the local oscillator phase. If phi=0, we are
        conditioning the state on the measured I-quadrature.
    detection_efficiency : scalar
        A number <= 1 specifying the detection efficiency. If equals 1, the
        detector is 100% efficient

    Returns
    -------
    rho_t : operator list
        A list of density matrices at each time given in time_list.
    wiener_incr : float array
        An array of gaussian distributed random numbers with variance equal
        to the time step and mean equal to 0. Used to reconstruct the output
        signal used to condition the quantum state.
    """

    #state must be a density matrix, should include a check and raise error if not of the correct type
    #tyoe can be checked with rho.type

    #dissipation free part of the hamiltonian
    def H_time(t):
        return H_drive_function(t)  #+ H_function(t)

    #Liouville-von Neumann contribution
    def Liouville_von_Neumann(rho, hamiltonian):
        return -1j * qt.commutator(hamiltonian, rho, kind='normal')

    #dissipation on Lindblad form
    def lindblad_diss(rho, dissipator):
        return dissipator * rho * dissipator.dag() - 0.5 * qt.commutator(
            dissipator.dag() * dissipator, rho, kind='anti')

    #stochastic part / diffusive part of the Ito SDE
    #measurement induced dynamics enters here
    def stochastic_part(rho, operator):
        LO_phase = np.exp(-1j * phi)
        sum1_part = LO_phase * operator * rho + rho * operator.dag(
        ) * LO_phase.conjugate()
        sum1_trace = qt.expect(rho, operator * LO_phase) + qt.expect(
            rho,
            operator.dag() * LO_phase.conjugate())
        return np.sqrt(detection_efficiency) * (sum1_part - sum1_trace * rho)

    #a small correction to avoid using derivatives of parameters (see algorithm for details)
    def Gamma_correction(rho, deterministic_part, stochastic_part, step_size):
        return rho + deterministic_part * step_size + stochastic_part * np.sqrt(
            step_size)

    #time step width
    width_t = np.abs(time_list[1] - time_list[0])

    #pre-generate a list of wiener increments
    #notice that the mean is zero
    #the standard deviation is sqrt(width_t), while variance is width_t
    #a list of gaussian distributed random numbers is generated
    wiener_incr = np.random.normal(0, np.sqrt(width_t), (len(time_list)))

    #preparing a list for the density matrix
    #each entry will correspond to the density matrix at the corresponding time step
    rho_t = list(range(len(time_list)))
    rho_t[0] = rho

    #The hamiltonian evolves independently of the state and can therefore be computed now
    #time independent part:
    H_no_time = H_function(0)
    H_list_original = [
        H_time(time_entry) + H_no_time for time_entry in time_list
    ]

    H_no_time_replacement = H_function_replace(0)
    H_list_replace = [
        H_time(time_entry) + H_no_time_replacement for time_entry in time_list
    ]

    z_bipartite = qt.tensor(qt.sigmaz(), qt.qeye(N))

    H_list = H_list_original

    for index, time_entry in enumerate(time_list):

        if index < len(time_list) - 1:

            dissipation_contribution = sum([
                lindblad_diss(rho_t[index], dissipator)
                for dissipator in dissipation_list
            ])

            drift = Liouville_von_Neumann(
                rho_t[index], H_list[index]) + dissipation_contribution

            diffusion = stochastic_part(rho_t[index], dissipation_list[0])

            gamma = Gamma_correction(rho_t[index], drift, diffusion, width_t)

            stoch_gamma = stochastic_part(gamma, dissipation_list[0])

            rho_t[index + 1] = rho_t[
                index] + drift * width_t + diffusion * wiener_incr[index]
            +0.5 * (stoch_gamma - diffusion) * (
                wiener_incr[index]**2 - width_t) / np.sqrt(width_t)

            if np.abs(qt.expect(rho_t[index + 1], z_bipartite)) > 0.95:
                H_list = H_list_replace

    return rho_t, wiener_incr
