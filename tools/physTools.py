"""

    physTools.py

    Conor O'Riordan
    Oct 2016

    Contains functions for doing all the actual physical modelling
    in the reconstruction, e.g. computing brightness profile and
    computing deflection angle field.

"""

import numpy as np
from scipy.special import hyp2f1
from mpmath import fp


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


def tessore(grid, g, f, m, a, trunc=False, r0=0.5):
    """
    Using Tessore's method (Tessore, 2016), calculates the deflection
    angle on the grid given a set of lens parameters in p.
    """
    # Convert our gamma to Tessore's t and angle to radians
    t = g - 1.0
    a *= - np.pi / 180.0

    x1, x2 = grid

    xx = x1 * np.cos(a) - x2 * np.sin(a)
    yy = x1 * np.sin(a) + x2 * np.cos(a)

    # Define the complex plane
    z = (xx + 1j * yy)  # / np.sqrt(f)

    # Define the elliptical and circular radial coordinates
    rootf = f ** 0.5
    rad = np.hypot(xx * f, yy)

    # Truncate the radial coordinates (truncates the mass distribution)
    # r0 *= rootf
    if trunc:
        r_ = rad * (rad < r0).astype('float') + r0 * (rad >= r0).astype('float')
    else:
        r_ = rad

    # Define terms for the angular part
    h1, h2, h3 = 0.5, 1.0 - t / 2.0, 2.0 - t / 2.0

    # Calculate normalisation from the EINSTEIN RADIUS
    norm = (1.0 / f) * (m ** 2.0)

    # Radial part
    radial = (1.0 / z) * ((m / r_) ** (t - 2.0))

    # Angular part
    h4 = ((1 - f ** 2.0) / f ** 2.0) * ((r_ ** 2.0) / (z ** 2.0))
    angular = hyp2f1(h1, h2, h3, h4)

    # Find the bad values in the angular part
    mask = np.where(np.log10(np.abs(angular)) > 0.0)

    # If bad values are present, correct them...
    if len(mask[0]) != 0:
        # Create a function from the mpmath hyp2f1 to be vectorized
        def mphyp(x):
            return fp.hyp2f1(h1, h2, h3, complex(x))

        # Vectorize function
        vec = np.vectorize(mphyp)

        # Recalculate bad values using mpmath
        angular[mask] = vec(h4[mask])

    # Deflection angle
    alpha = (- norm * radial * angular).conjugate()

    # Rotate the vector field
    a1, a2 = np.real(alpha * np.exp(1j * -a)), \
             np.imag(alpha * np.exp(1j * -a))

    # Return the transformed source plane coordinates
    return x1 + a1, x2 + a2


def tessore_double(grid, g1, g2, f, m1, m2, a, r0):
    """
    Calculates the deflection angle due to a double power law
    with separate slopes, broken at r0
    """

    b1 = m1
    # pref = (3.0 - g1) / (3.0 - g2)
    # rt = np.sqrt(f) * r0
    b2 = m2 # rt * (pref * (b1 / rt) ** (g1 - 1.0)) ** (1.0 / (g2 - 1.0))

    t11, t12 = tessore(grid, g1, f, m1, a, trunc=True, r0=r0)
    t21, t22 = tessore(grid, g2, f, m2, a, trunc=False)
    t31, t32 = tessore(grid, g2, f, m2, a, trunc=True, r0=r0)

    a1 = t11 + (t21 - t31)
    a2 = t12 + (t22 - t32)

    return a1, a2


def tessore_triple(grid, g1, g2, g3, f, m1, m2, m3, a, r1, r2):
    """
    Calculates the deflection angle due to a double power law
    with separate slopes, broken at r0
    """

    # For mass profiles with varying normalisation
    b1, b2, b3 = m1, m2, m3

    # # For mass profiles fixed by continuity
    # pref1 = (3.0 - g1) / (3.0 - g2)
    # rt1 = np.sqrt(f) * r1
    # b2 = rt1 * (pref1 * (b1 / rt1) ** (g1 - 1.0)) ** (1.0 / (g2 - 1.0))
    # pref2 = (3.0 - g2) / (3.0 - g3)
    # rt2 = np.sqrt(f) * r2
    # b3 = rt2 * (pref2 * (b2 / rt2) ** (g2 - 1.0)) ** (1.0 / (g3 - 1.0))

    t11, t12 = tessore(grid, g1, f, b1, a, trunc=True, r0=r1)
    t21, t22 = tessore(grid, g2, f, b2, a, trunc=True, r0=r2)
    t31, t32 = tessore(grid, g2, f, b2, a, trunc=True, r0=r1)
    t41, t42 = tessore(grid, g3, f, b3, a, trunc=False)
    t51, t52 = tessore(grid, g3, f, b3, a, trunc=True, r0=r2)

    a1 = t11 + (t21 - t31) + (t41 - t51)
    a2 = t12 + (t22 - t32) + (t42 - t52)

    return a1, a2


def tessore_switch(grid, g1, g2, g3, f, m1, m2, m3, a, r1, r2,
                   npow=1, trunc=False):
    """
    Chooses the number of power laws to use based on npow
    """
    if npow == 1:
        return tessore(grid, g1, f, m1, a, trunc=trunc, r0=r1)
    elif npow == 2:
        return tessore_double(grid, g1, g2, f, m1, m2, a, r1)
    elif npow == 3:
        return tessore_triple(grid, g1, g2, g3, f, m1, m2, m3, a, r1, r2)


def point_mass(grid, m):

    x1, x2 = grid

    r = np.hypot(x1, x2)
    phi = np.arctan2(x1, x2)

    # Calculate normalisation from the TOTAL MASS
    norm = 4.0 * m / r

    return x1 - norm * np.sin(phi), x2 - norm * np.cos(phi)