import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import gridspec
from matplotlib.pyplot import get_cmap
from tools.lensTools import pixels, deflection_switch, end_append
from tools.physTools import sersic, deflection
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.optimize import brenth
from scipy.interpolate import interp1d
from imp import load_source
from os import environ
from mpl_toolkits.axes_grid import make_axes_locatable
from matplotlib.pyplot import subplot

# Turn off divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')


class App:

    def __init__(self, master):
        """
        Starts the app and contains the main loop
        """

        self.txcol = '#f9f9f9'
        self.bgcol = '#2e3642'
        self.fgcol = '#525f72'

        # Main frame
        frame = tk.Frame(master, bg=self.bgcol)

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
        self.sliders_frame = tk.Frame(frame, bg=self.bgcol)
        self.sliders_frame.grid(row=0, column=1, padx=30)

        # Power law num. slider
        self.plaw_sliders_frame = tk.Frame(self.sliders_frame, bg=self.bgcol)
        self.plaw_sliders_frame.grid(row=0, column=0, padx=0, pady=20, ipadx=10, ipady=10, sticky='N')

        # Source parameter sliders
        self.source_sliders_frame = tk.Frame(self.sliders_frame, bg=self.bgcol)
        self.source_sliders_frame.grid(row=1, column=0, padx=0, ipadx=10, ipady=10, sticky='N')

        # Lens parameter sliders
        self.lens_sliders_frame = tk.Frame(self.sliders_frame, bg=self.bgcol)
        self.lens_sliders_frame.grid(row=1, column=1, padx=0, ipadx=10, ipady=10, sticky='N')

        # Lens parameter sliders
        self.lens_sliders_frame2 = tk.Frame(self.sliders_frame, bg=self.bgcol)
        self.lens_sliders_frame2.grid(row=1, column=2, padx=0, ipadx=10, ipady=10, sticky='N')

        # Image parameter frame
        self.image_frame = tk.Frame(frame, bg=self.bgcol)
        self.image_frame.grid(row=0, column=0)

        # Image parameter sliders
        self.image_sliders = tk.Frame(self.sliders_frame, bg=self.bgcol)
        self.image_sliders.grid(row=0, column=1, sticky='N', padx=0, pady=20, ipadx=10, ipady=10)
        self.image_sliders2 = tk.Frame(self.sliders_frame, bg=self.bgcol)
        self.image_sliders2.grid(row=0, column=2, sticky='N', padx=0, pady=20, ipadx=10, ipady=10)

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
        self.source_text = tk.Label(self.source_sliders_frame, text='Source Params', bg=self.bgcol, fg=self.txcol)
        self.source_sliders = [
            tk.Scale(self.source_sliders_frame, from_=smin[i], to=smax[i], label=slabel[i],
                     command=self.update, resolution=0.02, orient=tk.HORIZONTAL,
                     bg=self.bgcol, fg=self.txcol)
            for i in range(len(slabel))
        ]

        # Pack source sliders
        self.source_text.pack()
        for i, s in enumerate(self.source_sliders):
            s.set(sdef[i])      # Set default value
            s.pack()            # Pack sliders

        # Initialise power law slider ----------------------------------------------------------------------------------
        self.plaw_slider = tk.Scale(self.plaw_sliders_frame, label='No. P. Laws', bg=self.bgcol,
                                    from_=1, to=3, orient=tk.HORIZONTAL, resolution=1,
                                    command=self.update, fg=self.txcol)
        self.plaw_slider.set(self.p.npow)
        self.plaw_slider.pack()

        # Properties of lens sliders -----------------------------------------------------------------------------------
        llabel = ['Gamma 1', 'Ellipticty', 'Einstein Rad.', 'Position Angle', 'Inner Break']    # Parameter names
        lmin = [1.1, 0.0, 0.1, -180.0, 0.1]                                                     # Minimum values
        lmax = [2.98, 0.9, 3.0, 180.0, 3.0]                                                     # Maximum values
        ldef = [self.p.gmm1, 1.0 - self.p.axro, self.p.mss1, self.p.posa, self.p.rad1]          # Default values (from template file)
        lres = [0.02, 0.02, 0.02, 10.0, 0.02]

        # Initialise lens sliders
        self.lens_text = tk.Label(self.lens_sliders_frame, text='Single PLaw Params.', bg=self.bgcol, fg=self.txcol)
        self.lens_sliders = [
            tk.Scale(self.lens_sliders_frame, from_=lmin[i], to=lmax[i], label=llabel[i], bg=self.bgcol,
                     resolution=lres[i], orient=tk.HORIZONTAL, fg=self.txcol, command=self.update)
            for i in range(len(llabel))
        ]

        # Bind lens sliders to slow update on button release only and pack
        self.lens_text.pack()
        for i, l in enumerate(self.lens_sliders):
            l.set(ldef[i])
            l.pack()

        # Properties of lens 2 sliders ---------------------------------------------------------------------------------
        llabel = ['Gamma 2', 'Gamma 3', 'Outer Break']    # Parameter names
        lmin = [1.1, 1.1, 0.1]                                                     # Minimum values
        lmax = [2.98, 2.98, 3.0]                                                     # Maximum values
        ldef = [self.p.gmm2, self.p.gmm3, self.p.rad2]          # Default values (from template file)
        lres = [0.02, 0.02, 0.02]

        # Initialise lens sliders
        self.lens_text2 = tk.Label(self.lens_sliders_frame2, text='Multiple PLaw Params', bg=self.bgcol, fg=self.txcol)
        self.lens_sliders2 = [
            tk.Scale(self.lens_sliders_frame2, from_=lmin[i], to=lmax[i], label=llabel[i], bg=self.bgcol,
                     resolution=lres[i], orient=tk.HORIZONTAL, fg=self.txcol, command=self.update)
            for i in range(len(llabel))
        ]

        # Bind lens sliders to slow update on button release only and pack
        self.lens_text2.pack()
        for i, l in enumerate(self.lens_sliders2):
            l.set(ldef[i])
            l.pack()

        # Initialise image parameter sliders ---------------------------------------------------------------------------

        # SNR Level
        self.snr_slider = tk.Scale(self.image_sliders, from_=10.0, to=1000, label='SNR in Mask',
                                   resolution=10, orient=tk.HORIZONTAL, command=self.update_plots, bg=self.bgcol, fg=self.txcol)
        self.snr_slider.set(self.p.snr)
        self.snr_slider.pack()
        self.mask_bool = tk.BooleanVar(value=False)
        self.radii_bool = tk.BooleanVar(value=False)
        self.cc_bool = tk.BooleanVar(value=True)

        # Initialise image figure --------------------------------------------------------------------------------------
        self.fig = Figure(figsize=(6, 8))                                       # Open figure object
        gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax3 = self.fig.add_subplot(gs[1])
        # div = make_axes_locatable(self.ax1)
        # self.ax2 = div.append_axes('bottom', '1.5%', pad=0.0)

        self.fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0,
                                 hspace=0.0, wspace=0.0)      # Fill page with axis

        self.lens_canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.lens_canvas.show()

        # Bind mouse events in the image plane to the source sliders to
        # drag the source around with the mouse
        self.lens_canvas.mpl_connect('button_press_event', self.on_press)
        self.lens_canvas.mpl_connect('button_release_event', self.on_release)
        self.lens_canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.lens_canvas_w = self.lens_canvas.get_tk_widget()
        self.lens_canvas_w.grid(column=0, row=0)
        self.lens_canvas_w.pack(fill='both', expand=True)

        # Other variables to be init. later ----------------------------------------------------------------------------
        [self.src, self.img, self.alpha, self.press,
         self.n_level, self.mask, self.img_noise,
         self.caustic, self.critcurve, self.cbar] = [None] * 10

        # Perform first lens calculations
        self.update()

        # Mask button
        self.mask_button = tk.Checkbutton(self.image_sliders2, variable=self.mask_bool,
                                          onvalue=True, offvalue=False, justify='left', fg=self.txcol,
                                          text='Show Mask', command=self.update_plots, bg=self.bgcol)
        self.mask_button.pack()

        # Elliptical Radiii Button
        self.radii_button = tk.Checkbutton(self.image_sliders2, variable=self.radii_bool,
                                           onvalue=True, offvalue=False, justify='left', fg=self.txcol,
                                           text='Show Radii', command=self.update_plots, bg=self.bgcol)
        self.radii_button.pack()

        # Caustic button
        self.cc_button = tk.Checkbutton(self.image_sliders2, variable=self.cc_bool,
                                        activeforeground=self.txcol, disabledforeground=self.txcol,
                                        onvalue=True, offvalue=False, justify='left', fg=self.txcol,
                                        text='Show CC/Ca.', command=self.update_plots, bg=self.bgcol)
        self.cc_button.pack()

        # Pack the whole frame
        frame.pack()

    def on_press(self, event):

        if event.inaxes != self.ax1:
            return

        contains, attrd = self.ax1.contains(event)

        if not contains:
            return

        x0, y0 = self.source_sliders[0].get(), self.source_sliders[1].get()
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):

        if self.press is None:
            return
        if event.inaxes != self.ax1:
            return

        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        self.source_sliders[0].set(x0+dx)
        self.source_sliders[1].set(y0+dy)

        self.update()

    def on_release(self, event):

        self.press = None
        self.update()

    def update_plots(self, event=None):
        """
        Redraws the image plane frame after any update
        """

        # Image Plane plot ------------------------------------------------------------------------- #

        # Get the new noise level and mask
        self.n_level, self.mask = self.snr_set()

        # Add new noise level to the image
        self.img_noise = self.img + np.random.normal(0.0, self.n_level, self.img.shape)

        # Clear the previous iteration's plot
        self.ax1.clear()

        # Plot the main frame
        if self.plot_type.get() == 'img':
            cm = self.ax1.imshow(self.img_noise, extent=[-2, 2, -2, 2], vmin=-0.1, vmax=0.5,
                                 origin='lower', interpolation='none', cmap=get_cmap(self.cmap.get()))

        elif self.plot_type.get() == 'vec':
            dx = 5
            a1, a2 = self.alpha_convert()
            cm = self.ax1.quiver(self.pix[0][(dx//2)::dx, (dx//2)::dx],
                                 self.pix[1][(dx//2)::dx, (dx//2)::dx],
                                 a1[(dx//2)::dx, (dx//2)::dx],
                                 a2[(dx//2)::dx, (dx//2)::dx],
                                 np.hypot(a1[(dx//2)::dx, (dx//2)::dx], a2[(dx//2)::dx, (dx//2)::dx]))

        elif self.plot_type.get() == 'mag':
            a1, a2 = self.alpha_convert()
            cm = self.ax1.contourf(self.pix[0], self.pix[1], np.hypot(a1, a2), 10,
                                   vmin=0.5, vmax=1.5)

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

        # Add truncation radius:
        if self.plaw_slider.get() != 3:
            levels = [self.lens_sliders[4].get()]
        else:
            levels = [self.lens_sliders[4].get(), self.lens_sliders2[2].get()]

        self.ax1.contour(self.pix[0], self.pix[1],
                         np.hypot(self.pix[0] * (1.0 - self.lens_sliders[1].get()), self.pix[1]),
                         levels=levels, colors=self.detail_color, alpha=0.5)

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

        # Add the colorbar
        # self.cbar = self.fig.colorbar(cm, cax=self.ax2, orientation='horizontal')
        # self.cbar.set_ticks([])
        # self.ax2.xaxis.set_ticks_position('top')
        # self.ax2.xaxis.label.set_color(self.detail_color)
        # self.ax2.tick_params(axis='x', colors=self.detail_color)

        # Formatting
        self.ax1.set(xticks=[], yticks=[], xlim=[-2, 2], ylim=[-2, 2])
        self.ax1.axhline(0.0, color=self.detail_color, linestyle='-', alpha=0.5, lw=1.0)
        self.ax1.axvline(0.0, color=self.detail_color, linestyle='-', alpha=0.5, lw=1.0)

        # Deflection Angle Plot ------------------------------------------------------------------ #
        self.ax3.clear()

        # Calculate deflection angle along the x-axis
        x = self.pix[0][50, :]
        a1, a2 = self.alpha_convert()
        a = np.hypot(a1[50, :], a2[50, :])
        b = self.lens_sliders[2].get()

        # Calculate mass distribution
        k = kappa(np.abs(x),
                  b,
                  self.lens_sliders[0].get(),
                  self.lens_sliders2[0].get(),
                  self.lens_sliders2[1].get(),
                  self.lens_sliders[4].get() / (1.0 - self.lens_sliders[1].get()),
                  self.lens_sliders2[2].get() / (1.0 - self.lens_sliders[1].get()),
                  self.plaw_slider.get())

        self.ax3.plot(x, a, 'C0', label='$\\alpha(\\theta)$')
        self.ax3.plot(x, k, 'C1', label='$\\kappa(\\theta)$')

        self.ax3.legend(fontsize='x-small')
        self.ax3.tick_params(axis='y', direction='in', colors='w', pad=-20.0, labelsize=10)

        self.ax3.axvline(0.0, color=self.detail_color, linestyle='-', alpha=0.5, lw=1.0)

        self.ax3.set_facecolor(self.bgcol)
        self.ax3.axvline(b / (1.0 - self.lens_sliders[1].get()), linestyle='dotted', color='w', alpha=0.5)
        self.ax3.axvline(- b / (1.0 - self.lens_sliders[1].get()), linestyle='dotted', color='w', alpha=0.5)

        self.ax3.axvline(self.lens_sliders[4].get() / (1.0 - self.lens_sliders[1].get()), linestyle='dashed', color='w', alpha=0.5)
        self.ax3.axvline(- self.lens_sliders[4].get() / (1.0 - self.lens_sliders[1].get()), linestyle='dashed', color='w', alpha=0.5)
        if self.plaw_slider.get() == 3:
            self.ax3.axvline(self.lens_sliders2[2].get() / (1.0 - self.lens_sliders[1].get()), linestyle='dashed',
                             color='w', alpha=0.5)
            self.ax3.axvline(- self.lens_sliders2[2].get() / (1.0 - self.lens_sliders[1].get()), linestyle='dashed',
                             color='w', alpha=0.5)

        # if self.plaw_slider.get() == 1:
        #
        #     if self.lens_sliders[4].get() > 2.0:
        #         x = 1.0
        #     else:
        #         x = 0.5 * self.lens_sliders[4].get() / (1.0 - self.lens_sliders[1].get())
        #     self.ax3.text(x, 0.2, '$\gamma_1=%.2f$' % self.lens_sliders[0].get(),
        #                   ha='center', va='bottom', color='w', alpha=0.5)
        #
        # if self.plaw_slider.get() == 2:
        #
        #     if self.lens_sliders[4].get() > 2.0:
        #         x1 = 1.0
        #         x2 = 3.0
        #     else:
        #         x1 = 0.5 * self.lens_sliders[4].get() / (1.0 - self.lens_sliders[1].get())
        #         x2 = x1 + 0.5 * (2.0 / (1.0 - self.lens_sliders[1].get()) - x1)
        #
        #     self.ax3.text(x1, 0.2, '$\gamma_1=%.2f$' % self.lens_sliders[0].get(),
        #                   ha='center', va='bottom', color='w', alpha=0.5)
        #     self.ax3.text(x2, 0.2, '$\gamma_2=%.2f$' % self.lens_sliders2[0].get(),
        #                   ha='center', va='bottom', color='w', alpha=0.5)
        #
        self.ax3.set(xlim=[-2.0, 2.0], ylim=[0, 2.5],
                     yticks=[])

        self.lens_canvas.draw()

    def update(self, event=None):
        """
        Updates for faster functions e.g. changing source properties or SNR etc.
        """

        self.src = sersic(self.pix,
                          self.source_sliders[0].get(),
                          self.source_sliders[1].get(),
                          self.source_sliders[2].get(), self.p.srcb,
                          self.source_sliders[3].get())

        self.alpha = deflection_switch(self.pix,
                                       self.lens_sliders[0].get(),
                                       self.lens_sliders2[0].get(),
                                       self.lens_sliders2[1].get(),
                                       1.0 - self.lens_sliders[1].get(),
                                       self.lens_sliders[2].get(),
                                       self.lens_sliders[3].get(),
                                       self.lens_sliders[4].get(),
                                       self.lens_sliders2[2].get(),
                                       npow=self.plaw_slider.get(),
                                       trunc=True)

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
        x1_caus, x2_caus = deflection_switch((x1_crit, x2_crit),
                                             self.lens_sliders[0].get(),
                                             2.0,
                                             2.0,
                                             1.0 - self.lens_sliders[1].get(),
                                             self.lens_sliders[2].get(),
                                             self.lens_sliders[3].get(),
                                             self.lens_sliders[4].get(),
                                             3.0,
                                             npow=self.plaw_slider.get(),
                                             trunc=True)

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


def mass_continuity(b1, g1, g2, r0):
    """
    Calculates the normalisation of the second profile
    to maintain continuity at theta = r0.
    """
    return r0 * (((3.0 - g1) / (3.0 - g2)) * (b1 / r0) ** (g1 - 1.0)) ** (1.0 / (g2 - 1.0))


def kappa(x, b1, g1, g2, g3, r1, r2, npow):
    """
    Calculates a three power law mass distribution
    """

    b2 = mass_continuity(b1, g1, g2, r1)
    b3 = mass_continuity(b2, g2, g3, r2)

    k = lambda r, b, g: ((3.0 - g) / 2.0) * (b / r) ** (g - 1.0)

    if npow == 1:
        return (k(x, b1, g1) * (x < r1).astype('float') +
                0.0 * (x >= r1).astype('float')) * single_correction(True, r1, b1, g1)

    if npow == 2:
        return (k(x, b1, g1) * (x < r1).astype('float') +
                k(x, b2, g2) * (x >= r1).astype('float')) * double_correction(b1, b2, g1, g2, r1)

    if npow == 3:
        return (k(x, b1, g1) * (x < r1).astype('float') +
                k(x, b2, g2) * (x >= r1).astype('float') * (x < r2).astype('float') +
                k(x, b3, g3) * (x >= r2).astype('float')) * triple_correction(b1, b2, b3, g1, g2, g3, r1, r2)


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


root = tk.Tk()
root.title('Live Lens')
app = App(root)
root.mainloop()
