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
    z = (xx + 1j * yy) / np.sqrt(f)

    # Define the elliptical and circular radial coordinates
    rootf = f ** 0.5
    r = np.sqrt((xx * rootf) ** 2 + (yy / rootf) ** 2)

    # Truncate the radial coordinates (truncates the mass distribution)
    r0 *= rootf
    r_ = r * (r < r0).astype('float') + r0 * (r >= r0).astype('float')

    # Define terms for the angular part
    h1, h2, h3 = 0.5, 1.0 - t / 2.0, 2.0 - t / 2.0

    if trunc:

        # Calculate normalisation from the TOTAL MASS
        norm = (4.0 * f * m * (r0 ** (t - 2.0))) ** (1.0 / t)

        # Calculate the radial part
        radial = (norm ** 2 / (f * z)) * ((norm / r_) ** (t - 2))

        # Calculate the angular part
        h4 = ((1 - f ** 2) / f ** 2) * (r_ / z) ** 2

        angular = hyp2f1(h1, h2, h3, h4)

        # Find the bad values in the angular part
        mask = np.where(np.log10(np.abs(angular)) > 1.0)

        # # If bad values are present, correct them...
        if len(mask[0]) != 0:

            # Create a function from the mpmath hyp2f1 to be vectorized
            def mphyp(x):
                return fp.hyp2f1(h1, h2, h3, complex(x))

            # Vectorize function
            vec = np.vectorize(mphyp)

            # Recalculate bad values using mpmath
            angular[mask] = vec(h4[mask])

        # Assemble deflection angle
        alpha = (- radial * angular).conjugate()

    else:

        # Calculate normalisation from the EINSTEIN RADIUS
        norm = (1.0 / f) * (m ** 2.0)

        # Radial part
        radial = (1.0 / z) * ((m / r) ** (t - 2.0))

        # Angular part
        h4 = ((1 - f ** 2.0) / f ** 2.0) * ((r ** 2.0) / (z ** 2.0))
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


def point_mass(grid, m):

    x1, x2 = grid

    r = np.hypot(x1, x2)
    phi = np.arctan2(x1, x2)

    # Calculate normalisation from the TOTAL MASS
    norm = 4.0 * m / r

    return x1 - norm * np.sin(phi), x2 - norm * np.cos(phi)