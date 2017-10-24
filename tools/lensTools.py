"""

    lensTools.py

    Conor O'Riordan
    Oct 2016

    Contains tools for the forward lens modelling: constructing
    source and lens planes and storing properties of lenses
    in the Lens class.

"""
import sys
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pickle import load
from physTools import sersic, tessore, point_mass
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.interpolate import interp1d
from scipy.optimize import brenth


class Params:
    """
    A class collating parameters into a vector that can be used
    by emcee
    """

    def __init__(self, v):
        # Source parameters

        self.srcx = v[0]  # Source 1 x position (arcsec)
        self.srcy = v[1]  # Source 1 y position (arcsec)
        self.srcr = v[2]  # Source 1 Eff. Rad. (arcsec)
        self.srcb = v[3]  # Source 1 peak brightness
        self.srcm = v[4]

        # Lens parameters

        self.mass = v[5]  # Lens total mass OR Einstein radius
        self.gamm = v[6]  # Lens mass distribution slope
        self.axro = v[7]  # Lens axis ratio (1 - ellipticity)
        self.posa = v[8]  # Lens position angle

    def to_dict(self):
        return {
            'srcx': self.srcx,
            'srcy': self.srcy,
            'srcr': self.srcr,
            'srcb': self.srcb,
            'srcm': self.srcm,
            'mass': self.mass,
            'gamm': self.gamm,
            'axro': self.axro,
            'posa': self.posa
        }


class Lens:
    """
    Creates and stores the original observation of a source + images
    """

    def __init__(self, fname, plot=True):

        # Turn off divide by zero warnings
        np.seterr(divide='ignore', invalid='ignore')

        # Imports the input parameters from the pickled file
        with open(fname, 'r') as f:
            self.p = load(f)

        # Store pixel coordinates for this lens
        self.pix = pixels(self.p.pwd, self.p.wid, 0.0, self.p.n)

        # Construct the source plane
        self.src = source(self.p)

        # Construct the (noiseless) image plane
        self.img0 = model(self.p.vectorize, self.p)

        # Get the level of noise and the 2-sigma mask
        self.noise, self.mask = self.snr_set()

        # Add the noise to the image
        self.img = self.img0 + np.random.normal(0.0, self.noise, self.img0.shape)

        # Plot if necessary
        if plot:
            self.fig = self.plot()

    def plot(self):

        # Plot settings
        rcParams['font.size'] = 12
        cmap = plt.get_cmap('magma')
        fig, ax = plt.subplots(1, 2)

        # Plot source plane and image plane
        im0 = ax[0].imshow(self.src, interpolation='none', cmap=cmap,
                           extent=[-2.0, 2.0, -2.0, 2.0], origin='lower')
        im1 = ax[1].imshow(self.img, interpolation='none', cmap=cmap,
                           extent=[-2.0, 2.0, -2.0, 2.0], origin='lower')

        # Plot contours of elliptical radius on image plane

        a = - self.p.posa * np.pi / 180.0
        xx = self.pix[0] * np.cos(a) - self.pix[1] * np.sin(a)
        yy = self.pix[0] * np.sin(a) + self.pix[1] * np.cos(a)

        msk = ax[1].contour(self.pix[0], self.pix[1],
                            self.mask, levels=[0.0], colors='w', alpha=0.5)

        rootf = np.sqrt(self.p.axro)

        if self.p.trunc:
            cx2 = ax[1].contour(self.pix[0], self.pix[1],
                                np.hypot(xx * rootf, yy / rootf),
                                levels=[self.p.trrd * rootf], colors='w',
                                alpha=0.5)

        else:
            cx2 = ax[1].contour(self.pix[0], self.pix[1],
                                np.hypot(xx * self.p.axro, yy),
                                levels=np.linspace(0.0, 2.0, 11), colors='w',
                                alpha=0.1)

        # Plot Caustic
        if not self.p.trunc:
            ca = caustic(self.p.axro, self.p.mass, a=a)
            ax[0].plot(ca[1], ca[0], 'w', alpha=0.5)

        # Axis settings
        src_extent = source_zoom(self.p.srcr, self.p.srcx, self.p.srcy)
        ax[0].set_xlim(src_extent[:2])
        ax[0].set_ylim(src_extent[2:])
        ax[0].set_xlabel('$\\beta_1$ (")')
        ax[0].set_ylabel('$\\beta_2$ (")')
        ax[1].set_xlabel('$\\theta_1$ (")')
        ax[1].set_ylabel('$\\theta_2$ (")')
        ax[1].set_xlim([-2.0, 2.0])
        ax[1].set_ylim([-2.0, 2.0])

        # Plot labels on planes
        ax[0].text(src_extent[0] + 0.05,
                   src_extent[2] + 0.05,
                   'Source Plane', color='w', size=14)
        ax[1].text(-1.8, -1.8, 'Image Plane', color='w', size=14)

        ax[0].grid(False)
        ax[1].grid(False)

        # Add dashed lines through centre
        x = np.linspace(-4.0, 4.0, 101)
        y = np.zeros_like(x)
        x_ = x * np.cos(-a) - y * np.sin(-a)
        y_ = x * np.sin(-a) + y * np.cos(-a)
        x1_ = x * np.cos(-a + np.pi / 2.0) - y * np.sin(-a + np.pi / 2.0)
        y1_ = x * np.sin(-a + np.pi / 2.0) + y * np.cos(-a + np.pi / 2.0)

        for ai in ax:
            ai.plot(x_, y_, color='w', linestyle='dotted', alpha=0.5)
            ai.plot(x1_, y1_, color='w', linestyle='dotted', alpha=0.5)

        # Adjust axes
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.15)

        # Add colorbar
        cbar_ax = fig.add_axes([0.05, 0.07, 0.9, 0.03])
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Brightness')

        # Save
        fig.text(0.05, 0.92, '$\\varepsilon=%.2f$, $S_x=%.2f$' %
                 (1.0 - self.p.axro, self.p.srcx), size=16)
        fig.set_size_inches((16, 9.5))
        return fig

    def snr_set(self):
        """
        Finds the level of noise which sets the integrated SNR within
        the 2-sigma contours to the specified value, via an interpolation.
        """
        # Integrate signal for signal
        total_sig = self.img0.sum()

        # Set possible noise levels according to total signal
        levels = total_sig * np.logspace(-6, -1, 50)

        # Calculate the snr at all noise levels
        snrs = np.array([snr_find(self.img0 + np.random.normal(0.0, n, self.img0.shape), n)[0]
                         for n in levels])

        # Remove NaN values
        levels = levels[np.isfinite(snrs)]
        snrs = snrs[np.isfinite(snrs)]

        # Interpolate a function f(noise) = SNR(noise) - SNR_Target
        f = interp1d(levels, snrs - self.p.snr, kind='linear')

        # Find the root
        r = brenth(f, levels[0], levels[-1])

        # Return both the noise levels and the mask from the convolved image
        return r, snr_find(self.img0, r)[1]


def pixels(pix_width, fov_width, pos_ang, num_pix):
    """
    Creates a grid of pixels
    """

    # Convert to radians
    pos_ang = pos_ang * np.pi / 180.0
    limit = (pix_width * (num_pix - 1)) / (fov_width / 2.0)

    # Define 1D grid
    x = np.linspace(- limit, limit, num_pix)

    # Return 2D grid
    xx, yy = np.meshgrid(x, x)

    # Rotate coordinates
    xx_ = xx * np.cos(pos_ang) - yy * np.sin(pos_ang)
    yy_ = xx * np.sin(pos_ang) + yy * np.cos(pos_ang)

    return xx_, yy_


def source(p):
    """
    Creates the source plane by adaptively integrating
    """
    # Storage for different grid sizes
    x1, x2 = [], []

    # Loop over sub-pixelisation size and create
    # up to p.sub sub-pixelised grids
    for s in range(1, p.sub + 1):
        # Define 1D grid
        x = np.linspace(- (p.wid / 2.0) + (p.pwd / 2.0),
                        (p.wid / 2.0) - (p.pwd / 2.0), p.n * s)

        # Define 2D grid and reshape
        xx, yy = map(lambda x: x.reshape(p.n, s, p.n, s), np.meshgrid(x, x))

        # Add to list
        x1.append(xx)
        x2.append(yy)

    # Get the source profile for the lowest sub-pixelisation
    src = np.sqrt(sersic((x1[0], x2[0]), p.srcx, p.srcy,
                         p.srcr, p.srcb, p.srcm).mean(axis=3).mean(axis=1))

    # Find the level of detail (0-9) neccessary for each pixel
    msk = (np.array(np.ceil(p.sub * src / np.max(src)), dtype='int') - 1).clip(0, p.sub - 1)

    # Storage for the final source
    src = np.zeros(shape=(p.n, p.n))

    # Loop over the pixels
    for i in range(p.n):
        for j in range(p.n):
            # Get sub-coordinates within each pixel
            x = x1[msk[i, j]][i, :, j, :]
            y = x2[msk[i, j]][i, :, j, :]

            # Integrate over those coordinates and save
            src[i, j] = sersic((x, y), p.srcx, p.srcy, p.srcr, p.srcb, p.srcm).mean()

    return src


def model(theta, fixed_parameters):
    """
    Creates the image plane by adaptively integrating. Theta is the vector in
    the parameter space. Fixed paramaters come directly from the file
    """
    # Storage for different grid sizes
    x1, x2 = [], []

    # Store temporary parameters
    tp = Params(theta)

    # Loop over sub-pixelisation size and create
    # up to p.sub sub-pixelised grids
    for s in range(1, fixed_parameters.sub + 1):
        # Define 1D grid
        x = np.linspace(- (fixed_parameters.wid / 2.0) + (fixed_parameters.pwd / 2.0),
                          (fixed_parameters.wid / 2.0) - (fixed_parameters.pwd / 2.0),
                        fixed_parameters.n * s)

        # Define 2D grid and reshape
        xx, yy = map(lambda x: x.reshape(fixed_parameters.n, s,
                                         fixed_parameters.n, s),
                     np.meshgrid(x, x))

        # Add to list
        x1.append(xx)
        x2.append(yy)

    img = np.sqrt(
        sersic(
            tessore((x1[0], x2[0]),
                    tp.gamm, tp.axro, tp.mass, tp.posa,
                    trunc=fixed_parameters.trunc,
                    r0=fixed_parameters.trrd),
            tp.srcx, tp.srcy, tp.srcr, tp.srcb, tp.srcm
        ).mean(axis=3).mean(axis=1))

    # Find the level of detail (0-sub) neccessary for that pixel
    msk = np.clip(np.array(np.ceil(fixed_parameters.sub * img / np.max(img)),
                           dtype='int') - 1, 0, fixed_parameters.sub - 1)

    # Storage for the final image
    img = np.zeros(shape=(fixed_parameters.n, fixed_parameters.n))

    # Loop over the pixels
    for i in range(fixed_parameters.n):
        for j in range(fixed_parameters.n):
            # Get sub-coordinates within each pixel
            x = x1[msk[i, j]][i, :, j, :]
            y = x2[msk[i, j]][i, :, j, :]

            # Integrate over those coordinates and save
            alpha = tessore((x, y), tp.gamm, tp.axro, tp.mass, tp.posa,
                            trunc=fixed_parameters.trunc,
                            r0=fixed_parameters.trrd)
            img[i, j] = sersic(alpha, tp.srcx, tp.srcy, tp.srcr,
                               tp.srcb, tp.srcm).mean()

    return img


def caustic(f, b, a=0.0):

    theta = np.linspace(0.0, 2 * np.pi, 1000)
    delta = np.hypot(np.cos(theta), np.sin(theta) * f)

    f_ = np.sqrt(1.0 - f ** 2)

    y1 = b * (np.cos(theta) / delta - np.arcsinh(f_ * np.cos(theta) / f) / f_)
    y2 = b * (np.sin(theta) / delta - np.arcsin(f_ * np.sin(theta)) / f_)

    y1_ = y1 * np.cos(a) - y2 * np.sin(a)
    y2_ = y1 * np.sin(a) + y2 * np.cos(a)

    return y1_, y2_


def source_zoom(r, x, y, window=3.0):
    return [x - window * r,
            x + window * r,
            y - window * r,
            y + window * r]


def snr_find(image, nlevel, sig=2.0):
    """
    Calculates the integrated snr in within the 2-sigma contours.
    """
    # Initialise kernal and convolve
    g = Gaussian2DKernel(stddev=1.0)
    img1 = convolve(image, g, boundary='extend')

    # Take the 2-sigma contour of the convolved image
    mask = (img1 > sig * nlevel).astype('float')

    # Calculate snr of original image within the contours bof the convolved image
    return (mask * image).sum() / ((mask * nlevel ** 2.0).sum() ** 0.5), mask


def lens_plot(lens):

    xy = pixels(0.04, 4.0, 0.0, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot source plane and image plane
    im0 = ax1.imshow(lens.Source, interpolation='none',
                     extent=[-2.0, 2.0, -2.0, 2.0], origin='lower')
    im1 = ax2.imshow(lens.Image, interpolation='none',
                     extent=[-2.0, 2.0, -2.0, 2.0], origin='lower')

    msk = ax2.contour(xy[0], xy[1], lens.Mask, levels=[0.0], colors='w', alpha=0.5)

    cx2 = ax2.contour(xy[0], xy[1], np.hypot(xy[0] * (1.0 - lens.Ellipticity), xy[1]),
                      levels=np.linspace(0.0, 2.0, 11), colors='w', alpha=0.1)

    ca = caustic(1.0 - lens.Ellipticity, 1.0)
    ax1.plot(ca[1], ca[0], 'w', alpha=0.5)

    src_extent = source_zoom(0.10, lens.SourceX, lens.SourceY)
    ax1.set_xlim(src_extent[:2])
    ax1.set_ylim(src_extent[2:])
    ax1.set_xlabel('$\\beta_1$ (")')
    ax1.set_ylabel('$\\beta_2$ (")')
    ax2.set_xlabel('$\\theta_1$ (")')
    ax2.set_ylabel('$\\theta_2$ (")')
    ax2.set_xlim([-2.0, 2.0])
    ax2.set_ylim([-2.0, 2.0])

    # Add dashed lines through centre
    for a in np.linspace(0.0, np.pi, 21):

        x = np.linspace(-4.0, 4.0, 101)
        y = np.zeros_like(x)
        x_ = x * np.cos(-a) - y * np.sin(-a)
        y_ = x * np.sin(-a) + y * np.cos(-a)
        ax2.plot(x_, y_, color='w', linestyle='dotted', alpha=0.5)

    # Adjust axes
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.15)

    # Add colorbar
    cbar_ax = fig.add_axes([0.05, 0.07, 0.9, 0.03])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Brightness')
    fig.set_size_inches((16, 9.5))

    return fig
