"""
Module containing the Maxwell-Boltzmann PDF.
"""
import numpy as np
def maxwell(speed, kbt, mass=1.):
    """
    Calculate the Maxwell-Boltzmann probability density function (PDF) for a given speed.

    This function computes the probability density of finding a particle with a specific speed
    in an ideal gas, based on the Maxwell-Boltzmann distribution. It assumes the system is 
    isotropic and that the kinetic theory of gases applies.

    Args:
        speed (float or array): The speed(s) of the particle for which the PDF value is calculated.
        kbt (float): The product of Boltzmann's constant and the temperature of the gas.
        mass (float, optional): The mass of the particle. Default is 1.0. Should be
                                consistent with the units used for speed and kbt.

    Returns:
        float or array: The probability density value at the specified speed(s).
    """
    return speed*np.exp(-mass * np.power(speed, 2) / (2 * kbt))
