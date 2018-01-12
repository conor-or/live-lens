import Tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from tools.lensTools import pixels, tessore_switch, end_append
from tools.physTools import sersic, tessore
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.optimize import brenth
from scipy.interpolate import interp1d
from imp import load_source
from mpl_toolkits.axes_grid import make_axes_locatable

# Turn off divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')


class App:

    def __init__(self, master):
        """
        Starts the app and contains the main loop
        """

        # Main frame
        frame = tk.Frame(master)

        # Top level menu
        menubar = tk.Menu(master)
        menubar.add_command(label="Save", command=self.savefig)

        # Colourbar menu
        self.cmap = tk.StringVar(value='magma')
        self.detail_color = 'w'
        cmenu = tk.Menu(menubar, tearoff=0)
        cmenu.add_command(label="Viridis", command=lambda: self.cmap_set(c='viridis'))
        cmenu.add_command(label="Magma", command=lambda: self.cmap_set(c='magma'))
        cmenu.add_command(label="Greyscale", command=lambda: self.cmap_set(c='Greys'))
        cmenu.add_command(label="Jet", command=lambda: self.cmap_set(c='jet'))
        cmenu.add_command(label="Bone", command=lambda: self.cmap_set(c='bone_r'))
        menubar.add_cascade(label='Colour', menu=cmenu)

        # Plotting Menu
        self.plot_type = tk.StringVar(value='img')
        pmenu = tk.Menu(menubar, tearoff=0)
        pmenu.add_command(label="Image Plane", command=lambda: self.p_set(c='img'))
        pmenu.add_command(label="Deflection Angle (Vector)", command=lambda: self.p_set(c='vec'))
        pmenu.add_command(label="Deflection Angle (Magnitude)", command=lambda: self.p_set(c='mag'))
        menubar.add_cascade(label='Plotting', menu=pmenu)

        menubar.add_command(label="Quit", command=root.quit)
        master.config(menu=menubar)

        # Initialise frames --------------------------------------------------------------------------------------------

        # Sliders
        self.sliders_frame = tk.Frame(frame, bg='grey')
        self.sliders_frame.grid(row=0, column=1, padx=10)

        # Source parameter sliders
        self.source_sliders_frame = tk.Frame(self.sliders_frame)
        self.source_sliders_frame.grid(row=0, column=0, padx=5, ipadx=10, ipady=10, sticky='NW')

        # Lens parameter sliders
        self.lens_sliders_frame = tk.Frame(self.sliders_frame)
        self.lens_sliders_frame.grid(row=0, column=1, padx=5, ipadx=10, ipady=10, sticky='NW')

        # Lens 2 parameter sliders
        self.lens_sliders2_frame = tk.Frame(self.sliders_frame)
        self.lens_sliders2_frame.grid(row=0, column=2, padx=5, ipadx=10, ipady=10, sticky='NW')

        # Lens 3 parameter sliders
        self.lens_sliders3_frame = tk.Frame(self.sliders_frame)
        self.lens_sliders3_frame.grid(row=0, column=3, padx=5, ipadx=10, ipady=10, sticky='NW')

        # Image parameter frame
        self.image_frame = tk.Frame(frame, bg='grey')
        self.image_frame.grid(row=0, column=0)

        # Image parameter sliders
        self.image_sliders = tk.Frame(self.sliders_frame)
        self.image_sliders.grid(row=1, column=0, sticky='W', padx=5, pady=5, ipadx=10, ipady=10)

        # Load the default parameters
        self.p = load_source('', './templates/default.template').params()

        # Define pixel grid
        self.pix = pixels(self.p.pwd, self.p.wid, 0.0, self.p.n)

        # Properties of source parameter sliders
        slabel = ['X Pos.', 'Y Pos.', 'Eff. Radius', 'Sersic Index']        # Parameter names
        smin = [-0.8, -0.8, 0.02, 0.5]                                      # Minimum values
        smax = [0.8, 0.8, 0.5, 4.0]                                         # Maximum values
        sdef = [self.p.srcx, self.p.srcy, self.p.srcr, self.p.srcm]         # Default values (from template file)

        # Initialise source sliders  -----------------------------------------------------------------------------------
        self.source_text = tk.Label(self.source_sliders_frame, text='Source Params')
        self.source_sliders = [
            tk.Scale(self.source_sliders_frame, from_=smin[i], to=smax[i], label=slabel[i],
                     command=self.update_fast, resolution=0.02, orient=tk.HORIZONTAL)
            for i in range(len(slabel))
        ]

        # Pack source sliders
        self.source_text.pack()
        for i, s in enumerate(self.source_sliders):
            s.set(sdef[i])      # Set default value
            s.pack()            # Pack sliders

        # Properties of lens sliders -----------------------------------------------------------------------------------
        llabel = ['Gamma', 'Ellipticty', 'Total Mass', 'Position Angle']    # Parameter names
        lmin = [1.1, 0.0, 0.1, -np.pi]                                      # Minimum values
        lmax = [2.99, 0.9, 3.0, np.pi]                                      # Maximum values
        ldef = [self.p.gmm1, 1.0 - self.p.axro, self.p.mss1, self.p.posa]   # Default values (from template file)

        # Initialise lens sliders
        self.lens_text = tk.Label(self.lens_sliders_frame, text='Inner Params')
        self.lens_sliders = [
            tk.Scale(self.lens_sliders_frame, from_=lmin[i], to=lmax[i], label=llabel[i],
                     resolution=0.02, orient=tk.HORIZONTAL)
            for i in range(len(llabel))
        ]

        # Bind lens sliders to slow update on button release only and pack
        self.lens_text.pack()
        for i, l in enumerate(self.lens_sliders):
            l.bind("<ButtonRelease-1>", self.update_slow)
            l.set(ldef[i])
            l.pack()

        # Properties of lens 2 sliders ---------------------------------------------------------------------------------
        llabel = ['Gamma', 'Total Mass', 'Break Radius']
        lmin = [1.10, 0.1, 0.1]                                      # Minimum values
        lmax = [2.99, 3.0, 3.0]                                      # Maximum values
        ldef = [self.p.gmm2, self.p.mss2, self.p.rad1]               # Default values (from template file)

        # Initialise lens sliders
        self.lens_text2 = tk.Label(self.lens_sliders2_frame, text='Middle Params')
        self.lens_sliders2 = [
            tk.Scale(self.lens_sliders2_frame, from_=lmin[i], to=lmax[i], label=llabel[i],
                     resolution=0.02, orient=tk.HORIZONTAL)
            for i in range(len(llabel))]

        self.lens_text2.pack()
        for i, l in enumerate(self.lens_sliders2):
            l.bind("<ButtonRelease-1>", self.update_slow)
            l.set(ldef[i])
            l.pack()

        # Properties of lens 3 sliders ---------------------------------------------------------------------------------
        llabel = ['Gamma', 'Total Mass', 'Break Radius']
        lmin = [1.10, 0.1, 0.1]                                      # Minimum values
        lmax = [2.99, 3.0, 3.0]                                      # Maximum values
        ldef = [self.p.gmm3, self.p.mss3, self.p.rad2]               # Default values (from template file)

        # Initialise lens sliders
        self.lens_text3 = tk.Label(self.lens_sliders3_frame, text='Outer Params')
        self.lens_sliders3 = [
            tk.Scale(self.lens_sliders3_frame, from_=lmin[i], to=lmax[i], label=llabel[i],
                     resolution=0.02, orient=tk.HORIZONTAL)
            for i in range(len(llabel))]

        self.lens_text3.pack()
        for i, l in enumerate(self.lens_sliders3):
            l.bind("<ButtonRelease-1>", self.update_slow)
            l.set(ldef[i])
            l.pack()

        # Initialise image parameter sliders ---------------------------------------------------------------------------

        # SNR Level
        self.snr_slider = tk.Scale(self.image_sliders, from_=10.0, to=300, label='SNR in Mask',
                                   resolution=1.0, orient=tk.HORIZONTAL, command=self.update_plots)
        self.snr_slider.set(self.p.snr)
        self.snr_slider.pack()
        self.mask_bool = tk.BooleanVar(value=False)
        self.radii_bool = tk.BooleanVar(value=False)
        self.cc_bool = tk.BooleanVar(value=True)

        # Initialise image figure --------------------------------------------------------------------------------------
        self.fig = Figure(figsize=(6, 6.2))                                       # Open figure object
        self.ax1 = self.fig.add_subplot(111)
        div = make_axes_locatable(self.ax1)
        self.ax2 = div.append_axes('bottom', '3%', pad=0.0)

        self.fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)      # Fill page with axis

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.show()
        self.canvas_w = self.canvas.get_tk_widget()
        self.canvas_w.grid(column=0, row=0)
        self.canvas_w.pack(fill='both', expand=True)

        # Other variables to be init. later ----------------------------------------------------------------------------
        [self.src, self.img, self.alpha,
         self.n_level, self.mask, self.img_noise,
         self.caustic, self.critcurve] = [None] * 8

        # # Continuity button
        self.cont_bool = tk.BooleanVar(value=True)
        self.cont_button = tk.Checkbutton(self.lens_sliders2_frame, variable=self.cont_bool,
                                          onvalue=True, offvalue=False, justify='left',
                                          text='Mass Continuity', command=self.update_slow)
        self.cont_button.pack()

        # Perform first lens calculations
        self.update_slow()

        # Mask button
        self.mask_button = tk.Checkbutton(self.image_sliders, variable=self.mask_bool,
                                          onvalue=True, offvalue=False, justify='left',
                                          text='Show Mask', command=self.update_plots)
        self.mask_button.pack()

        # Elliptical Radiii Button
        self.radii_button = tk.Checkbutton(self.image_sliders, variable=self.radii_bool,
                                           onvalue=True, offvalue=False, justify='left',
                                           text='Show Radii', command=self.update_plots)
        self.radii_button.pack()

        # Caustic button
        self.cc_button = tk.Checkbutton(self.image_sliders, variable=self.cc_bool,
                                        onvalue=True, offvalue=False, justify='left',
                                        text='Show CC/Ca.', command=self.update_plots)
        self.cc_button.pack()

        # Pack the whole frame
        frame.config(bg='grey')
        frame.pack()

    def update_plots(self, event=None):
        """
        Redraws the image plane frame after any update
        """

        # Get the new noise level and mask
        self.n_level, self.mask = self.snr_set()

        # Add new noise level to the image
        self.img_noise = self.img + np.random.normal(0.0, self.n_level, self.img.shape)

        # Clear the previous iteration's plot
        self.ax1.clear()

        # Plot the main frame
        if self.plot_type.get() == 'img':
            cm = self.ax1.imshow(self.img_noise, extent=[-2, 2, -2, 2],
                                 origin='lower', interpolation='none', cmap=get_cmap(self.cmap.get()))

        elif self.plot_type.get() == 'vec':
            dx = 5
            a1, a2 = self.alpha_convert()
            cm = self.ax1.quiver(self.pix[0][(dx/2)::dx, (dx/2)::dx],
                                 self.pix[1][(dx/2)::dx, (dx/2)::dx],
                                 a1[(dx/2)::dx, (dx/2)::dx],
                                 a2[(dx/2)::dx, (dx/2)::dx],
                                 np.hypot(a1[(dx/2)::dx, (dx/2)::dx], a2[(dx/2)::dx, (dx/2)::dx]))

        elif self.plot_type.get() == 'mag':
            a1, a2 = self.alpha_convert()
            cm = self.ax1.contourf(self.pix[0], self.pix[1], np.hypot(a1, a2), extent=[-2, 2, -2, 2],
                                  origin='lower', interpolation='none', cmap=get_cmap(self.cmap.get()))

        else:
            cm = None

        # Add the mask
        if self.mask_bool.get():
            self.ax1.contour(self.pix[0], self.pix[1], self.mask,
                             levels=[0.0], colors=self.detail_color, alpha=0.2)

        # Add elliptical radii
        if self.radii_bool.get():
            self.ax1.contour(self.pix[0], self.pix[1],
                             np.hypot(self.pix[0] * (1.0 - self.lens_sliders[1].get()), self.pix[1]),
                             levels=np.linspace(0.0, 4.0, 21), colors=self.detail_color, alpha=0.2)

        # Plot the source position (x)
        self.ax1.plot(self.source_sliders[0].get(),
                      self.source_sliders[1].get(),
                      'x', color=self.detail_color)

        # Make a circle the size of source's eff. rad.
        source_circle = circle_coords(self.source_sliders[2].get(),
                                      self.source_sliders[0].get(),
                                      self.source_sliders[1].get())

        # Plot source radius
        self.ax1.plot(source_circle[0], source_circle[1], color=self.detail_color)

        # Plot the caustic and critical curve
        if self.cc_bool.get():
            self.ax1.plot(self.caustic[0], self.caustic[1], ':', color=self.detail_color, lw=1.0)
            self.ax1.plot(self.critcurve[0], self.critcurve[1], ':', color=self.detail_color, lw=1.0)

        # Plot the break radii
        self.ax1.contour(self.pix[0], self.pix[1],
                         np.hypot(self.pix[0] * (1.0 - self.lens_sliders[1].get()), self.pix[1]),
                         levels=[self.lens_sliders2[2].get(), self.lens_sliders3[2].get()],
                         colors=self.detail_color, alpha=0.2)

        # Add the colorbar
        self.cbar = self.fig.colorbar(cm, cax=self.ax2, orientation='horizontal')
        self.ax2.xaxis.set_ticks_position('top')
        self.ax2.xaxis.label.set_color(self.detail_color)
        self.ax2.tick_params(axis='x', colors=self.detail_color)

        # Formatting
        self.ax1.set(xticks=[], yticks=[], xlim=[-2, 2], ylim=[-2, 2])
        self.ax1.axhline(0.0, color=self.detail_color, linestyle='-', alpha=0.5, lw=1.0)
        self.ax1.axvline(0.0, color=self.detail_color, linestyle='-', alpha=0.5, lw=1.0)

        self.canvas.draw()

    def update_fast(self, event=None):
        """
        Updates for faster functions e.g. changing source properties or SNR etc.
        """

        self.src = sersic(self.pix,
                          self.source_sliders[0].get(),
                          self.source_sliders[1].get(),
                          self.source_sliders[2].get(), self.p.srcb,
                          self.source_sliders[3].get())

        self.img = sersic(self.alpha,
                          self.source_sliders[0].get(),
                          self.source_sliders[1].get(),
                          self.source_sliders[2].get(), self.p.srcb,
                          self.source_sliders[3].get())

        self.update_plots()

    def update_slow(self, event=None):
        """
        Updates for slow functions e.g. changing any of the lens parameters.
        """

        p = self.p_dict()
        if self.cont_bool.get():
            # # For mass profiles fixed by continuity
            pref1 = (3.0 - p['gmm1']) / (3.0 - p['gmm2'])
            rt1 = np.sqrt(p['axro']) * p['rad1']
            self.m2 = rt1 * (pref1 * (p['mss1'] / rt1) ** (p['gmm1'] - 1.0)) ** (1.0 / (p['gmm2'] - 1.0))
            pref2 = (3.0 - p['gmm2']) / (3.0 - p['gmm3'])
            rt2 = np.sqrt(p['axro']) * p['rad2']
            self.m3 = rt2 * (pref2 * (p['mss2'] / rt2) ** (p['gmm2'] - 1.0)) ** (1.0 / (p['gmm3'] - 1.0))
            self.lens_sliders2[1].set(self.m2)
            self.lens_sliders3[1].set(self.m3)

        else:
            self.m2 = self.lens_sliders2[1].get()
            self.m3 = self.lens_sliders3[1].get()

        self.src = sersic(self.pix,
                          self.source_sliders[0].get(),
                          self.source_sliders[1].get(),
                          self.source_sliders[2].get(), self.p.srcb,
                          self.source_sliders[3].get())

        self.alpha = tessore_switch(self.pix,
                                    self.lens_sliders[0].get(),
                                    self.lens_sliders2[0].get(),
                                    self.lens_sliders3[0].get(),
                                    1.0 - self.lens_sliders[1].get(),
                                    self.lens_sliders[2].get(),
                                    self.m2,
                                    self.m3,
                                    180.0 * self.lens_sliders[3].get() / np.pi,
                                    self.lens_sliders2[2].get(),
                                    self.lens_sliders3[2].get(), npow=3)

        self.img = sersic(self.alpha,
                          self.source_sliders[0].get(),
                          self.source_sliders[1].get(),
                          self.source_sliders[2].get(), self.p.srcb,
                          self.source_sliders[3].get())

        # Find the caustic and critical curve
        self.caustic, self.critcurve = self.caus_crit()
        self.update_plots()

    def snr_set(self):
        """
        Finds the level of noise which sets the integrated SNR within
        the 2-sigma contours to the specified value, via an interpolation.
        """
        # Integrate signal for signal
        total_sig = self.img.sum()

        # Set possible noise levels according to total signal
        levels = total_sig * np.logspace(-6, -1, 50)

        # Calculate the snr at all noise levels
        snrs = np.array([snr_find(self.img + np.random.normal(0.0, n, self.img.shape), n)[0]
                         for n in levels])

        # Remove NaN values
        levels = levels[np.isfinite(snrs)]
        snrs = snrs[np.isfinite(snrs)]

        # Interpolate a function f(noise) = SNR(noise) - SNR_Target
        f = interp1d(levels, snrs - self.snr_slider.get(), kind='linear')

        # Find the root
        r = brenth(f, levels[0], levels[-1])

        # Return both the noise levels and the mask from the convolved image
        return r, snr_find(self.img, r)[1].copy()

    def p_dict(self):

        return {
            'gmm1': self.lens_sliders[0].get(),
            'gmm2': self.lens_sliders2[0].get(),
            'gmm3': self.lens_sliders[0].get(),
            'axro': 1.0 - self.lens_sliders[1].get(),
            'mss1': self.lens_sliders[2].get(),
            'mss2': self.lens_sliders2[1].get(),
            'mss3': self.lens_sliders[2].get(),
            'posa': self.lens_sliders[3].get(),
            'rad1': self.lens_sliders2[2].get(),
            'rad2': 2.0, 'npow': 2
        }

    def alpha_convert(self):

        return self.alpha[0] - self.pix[0], self.alpha[1] - self.pix[1]

    def savefig(self):

        self.fig.savefig('./lens.png')

    def cmap_set(self, c='viridis'):

        self.cmap.set(c)

        if (c == 'Greys') or (c == 'bone_r'):
            self.detail_color = 'k'
        else:
            self.detail_color = 'w'

        self.update_plots()

    def p_set(self, c='viridis'):

        self.plot_type.set(c)
        if c == 'vec':
            self.detail_color = 'k'
        else:
            self.detail_color = 'w'
        self.update_plots()

    def caus_crit(self):
        """
        Given a deflection angle field and a set of lens parameters,
        numerically calculates the caustic and critical curves.
        """

        x1, x2 = self.pix
        a1_, a2_ = self.alpha_convert()
        a1, a2 = a1_ + 2 * x1, a2_ + 2 * x2

        # Get separation and calculate gradient
        dx = float(x1[0, 1] - x1[0, 0])
        a1_1 = np.gradient(a1, dx, axis=1)
        a2_2 = np.gradient(a2, dx, axis=0)
        a1_2 = np.gradient(a1, dx, axis=0)

        # Get determinant of Jac. and find zeros
        det_a = (1.0 - a1_1) * (1.0 - a2_2) - a1_2 ** 2
        f = (det_a > 0.0).astype('float') * (np.hypot(x1, x2) > 0.1).astype('float')
        det_mask = (np.gradient(f, axis=0) * np.gradient(f, axis=1)) ** 2 > 0.0

        # Get coordinates of zeros and transorm back to source plane
        # via deflection angle
        x1_crit, x2_crit = x1[det_mask], x2[det_mask]
        x1_caus, x2_caus = tessore_switch((x1_crit, x2_crit),
                                          self.lens_sliders[0].get(),
                                          self.lens_sliders2[0].get(),
                                          self.lens_sliders2[0].get(),
                                          1.0 - self.lens_sliders[1].get(),
                                          self.lens_sliders[2].get(),
                                          self.m2,
                                          self.m3,
                                          180.0 * self.lens_sliders[3].get() / np.pi,
                                          self.lens_sliders2[2].get(),
                                          self.lens_sliders3[2].get(), npow=3)

        # Sort the arrays so line plots are in the correct order
        ca_sort = np.argsort(np.arctan2(x1_caus, x2_caus))
        ca_sorted = end_append(x1_caus[ca_sort]), end_append(x2_caus[ca_sort])
        cc_sort = np.argsort(np.arctan2(x1_crit, x2_crit))
        cc_sorted = end_append(x1_crit[cc_sort]), end_append(x2_crit[cc_sort])

        return ca_sorted, cc_sorted


def circle_coords(r, x0, y0):

    t = np.linspace(0.0, 2 * np.pi, 100)
    x = (r * np.cos(t) - r * np.sin(t)) / np.sqrt(2) + x0
    y = (r * np.sin(t) + r * np.cos(t)) / np.sqrt(2) + y0

    return x, y


def caustic(f, b, a=0.0):

    theta = np.linspace(0.0, 2 * np.pi, 1000)
    delta = np.hypot(np.cos(theta), np.sin(theta) * f)

    f_ = np.sqrt(1.0 - f ** 2)

    y1 = b * (np.cos(theta) / delta - np.arcsinh(f_ * np.cos(theta) / f) / f_)
    y2 = b * (np.sin(theta) / delta - np.arcsin(f_ * np.sin(theta)) / f_)

    y1_ = y1 * np.cos(a) - y2 * np.sin(a)
    y2_ = y1 * np.sin(a) + y2 * np.cos(a)

    return y1_, y2_


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


root = tk.Tk()
root.title('live-lens')
app = App(root)
root.mainloop()
