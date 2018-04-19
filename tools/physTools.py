"""

    physTools.py

    Conor O'Riordan
    Oct 2016

    Contains functions for doing all the actual physical modelling
    in the reconstruction, e.g. computing brightness profile and
    computing deflection angle field.

"""

import numpy as np


# Define the normalisation constant for the Sersic profile
def sersic_b(m):

    return (2 * m) - (1.0 / 3.0) + (4.0 / (405.0 * m)) + \
           (46.0 / (25515.0 * m ** 2)) + (131.0 / (148174.0 * m ** 3))


def sersic(grid, srcx, srcy, srcr, srcb, srcm):
    """
    Creates a Sersic source centered on (srcx, srcy)
    with a radius srcr, peak brightness srcb and Sersuc index m.
    """

    # Shift coordinates with source at origin and scale by axis ratio
    x1 = (grid[0] - srcx)
    x2 = (grid[1] - srcy)

    # Convert to radial coordinates
    r = np.hypot(x1, x2)

    # Call the sersic profile
    return srcb * np.exp(- sersic_b(srcm) * (r / srcr) ** (1.0 / srcm))


def deflection_angular(z, g, n=30):
    """
    Finds the angular part of the deflection angle.
    """

    # Define z'
    z_ = (1.0 - np.sqrt(1.0 - z)) / 2.0
    
    # Set a_0
    a_n = [1.0]
    
    # Storage for results
    omega = np.zeros_like(z)

    # Loop over terms
    for i in range(n):
        
        # Add current term
        omega += a_n[i] * z_ ** i
        
        # Calculate and store next coefficient
        ratio = (2.0 * i - 2.0 * g + 6.0) / (2.0 * i - g + 5.0)
        a_n.append(ratio * a_n[i])

    return omega


def deflection(grid, g, f, m, a, trunc=False, r0=0.5, c=1.0, n=30):
    """
    Find the deflection angle via a sum over angular coefficients
    """

    # Define complex plane and rotate into mass frame
    x1, x2 = grid

    # Rotate into mass frame
    a *= - np.pi / 180.0
    xx = x1 * np.cos(a) - x2 * np.sin(a)
    yy = x1 * np.sin(a) + x2 * np.cos(a)
    z = xx + 1j * yy

    # Define elliptical radius
    r = np.hypot(xx * f, yy)
    
    # Truncate the radial coordinates (truncates the mass distribution)
    if trunc:
        r_ = r * (r < r0).astype('float') + r0 * (r >= r0).astype('float')
    else:
        r_ = r
    
    # Define z'
    z_ = (r_ ** 2 / z ** 2) * (1.0 - f ** 2) / (f ** 2)

    # Calculate angular and radial parts
    rad_part = c * (m ** 2 / (f * z)) * (m / r_) ** (g - 3.0)
    ang_part = deflection_angular(z_, g, n=n)

    # Multiply parts and take complex conjugate
    alpha = (- rad_part * ang_part).conjugate()
    
    # Rotate the vector field
    a1, a2 = np.real(alpha * np.exp(1j * -a)), \
             np.imag(alpha * np.exp(1j * -a))
    
    return x1 + a1, x2 + a2


def deflection_double(grid, g1, g2, f, m1, a, r0):
    """
    Calculates the deflection angle due to a double power law
    with separate slopes, broken at r0.
    """

    # Get the normalisation for the second profile such that the
    # profiles are continuous across r0
    m2 = mass_continuity(m1, g1, g2, r0)

    # Get the correction to the whole profile such that the Einstein
    # radius maintains its original definition
    correction = double_correction(m1, m2, g1, g2, r0)

    # Calculate the three components
    a11, a12 = deflection(grid, g1, f, m1, a, trunc=True,  r0=r0, c=correction)
    a21, a22 = deflection(grid, g2, f, m2, a, trunc=False, r0=r0, c=correction)
    a31, a32 = deflection(grid, g2, f, m2, a, trunc=True,  r0=r0, c=correction)

    # Make the composite profile
    return a11 + (a21 - a31), a12 + (a22 - a32)


def deflection_triple(grid, g1, g2, g3, f, m1, a, r1, r2):
    """
    Calculates the deflection angle due to a triple power law
    with separate slopes, broken at r1 and r2.
    """

    # Get the normalisation for the second profile such that the
    # profiles are continuous across r0
    m2 = mass_continuity(m1, g1, g2, r1)
    m3 = mass_continuity(m2, g2, g3, r2)

    # Get the correction to the whole profile such that the Einstein
    # radius maintains its original definition
    correction = triple_correction(m1, m2, m3, g1, g2, g3, r1, r2)

    # Calculate the three components

    # Profile from 0 to r1
    a11, a12 = deflection(grid, g1, f, m1, a, trunc=True, r0=r1, c=correction)

    # From r1 to r2
    a21, a22 = deflection(grid, g2, f, m2, a, trunc=True, r0=r2, c=correction)
    a31, a32 = deflection(grid, g2, f, m2, a, trunc=True, r0=r1, c=correction)

    # From r2 to infty
    a41, a42 = deflection(grid, g3, f, m3, a, trunc=False, c=correction)
    a51, a52 = deflection(grid, g3, f, m3, a, trunc=True, r0=r2, c=correction)

    # Make the composite profile
    return a11 + (a21 - a31) + (a41 - a51), a12 + (a22 - a32) + (a42 - a52)


def deflection_switch(grid, g1, g2, g3, f, m1, a, r1, r2, npow=1, trunc=False):
    """
    Handles the option of different number of power laws.
    """

    # Single power law (including truncated)
    if npow == 1:
        correction = single_correction(trunc, r1, m1, g1)
        return deflection(grid, g1, f, m1, a, trunc=trunc, r0=r1, c=correction)

    # Double power law
    elif npow == 2:
        return deflection_double(grid, g1, g2, f, m1, a, r0=r1)

    # Triple power law
    else:
        return deflection_triple(grid, g1, g2, g3, f, m1, a, r1, r2)


def single_correction(t, r, m, g):
    """
    Calculates the correction to the deflection angle
    if using a truncated profile. This ensures that b keeps
    the definition it has in an untruncated profile.
    """

    if not t:
        c = 1.0
    elif r > m:
        c = 1.0
    else:
        c = ((m / r) ** (3.0 - g))

    return c


def double_correction(b1, b2, g1, g2, r0):
    """
    Calculates the correction to the double mass profile
    such that the Einstein radius maintains its original
    definition.

    """
    def f(a, b, c):
        return (a ** (c - 1.0)) * (b ** (3.0 - c))

    return (b1 ** 2) / (f(b1, r0, g1) + f(b2, b1, g2) - f(b2, r0, g2))


def triple_correction(b1, b2, b3, g1, g2, g3, r1, r2):
    f = lambda a, b, c: (a ** (c - 1.0)) * (b ** (3.0 - c))

    return (b1 ** 2) / (f(b1, r1, g1) + f(b2, r2, g2) - f(b2, r1, g2) + f(b3, b1, g3) - f(b3, r2, g3))


def mass_continuity(b1, g1, g2, r0):
    """
    Calculates the normalisation of the second profile
    to maintain continuity at theta = r0.
    """
    return r0 * (((3.0 - g1) / (3.0 - g2)) * (b1 / r0) ** (g1 - 1.0)) ** (1.0 / (g2 - 1.0))
