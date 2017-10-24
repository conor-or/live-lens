import Tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap
from tools.lensTools import pixels
from tools.physTools import sersic, tessore
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.optimize import brenth
from scipy.interpolate import interp1d
from imp import load_source

# Turn off divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')


class App:

    def __init__(self, master):

        # Main frame
        frame = tk.Frame(master)

        self.sliders_frame = tk.Frame(frame, bg='grey')
        self.sliders_frame.grid(row=0, column=1)
        self.source_sliders_frame = tk.Frame(self.sliders_frame, bg='grey')
        self.source_sliders_frame.grid(row=0, column=0)
        self.lens_sliders_frame = tk.Frame(self.sliders_frame)
        self.lens_sliders_frame.grid(row=0, column=1)
        self.image_frame = tk.Frame(frame, bg='grey')
        self.image_frame.grid(row=0, column=0)
        self.image_sliders = tk.Frame(self.sliders_frame)
        self.image_sliders.grid(row=1, column=0)

        # Load default parameters
        self.p = load_source('', './templates/default.template').params()

        # Properties of source parameter sliders
        slabel = ['X Pos.', 'Y Pos.', 'Eff. Rad.', 'Sersic Index']
        smin = [-0.5, -0.5, 0.02, 0.5]
        smax = [0.5, 0.5, 0.5, 4.0]
        sdef = [self.p.srcx, self.p.srcy, self.p.srcr, self.p.srcm]

        # Initialise source sliders
        self.source_sliders = [
            tk.Scale(self.source_sliders_frame, from_=smin[i], to=smax[i], label=slabel[i],
                     command=self.update_fast, resolution=0.02, orient=tk.HORIZONTAL)
            for i in range(len(slabel))
        ]

        # Pack source sliders
        for i, s in enumerate(self.source_sliders):
            s.set(sdef[i])
            s.pack()

        # Properties of lens sliders
        llabel = ['Gamma', 'Ellipticty', 'Total Mass', 'Position Angle']
        lmin = [1.1, 0.0, 0.1, -np.pi]
        lmax = [2.99, 0.9, 3.0, np.pi]
        ldef = [self.p.gamm, 1.0 - self.p.axro, self.p.mass, self.p.posa]

        # Initialise lens sliders
        self.lens_sliders = [
            tk.Scale(self.lens_sliders_frame, from_=lmin[i], to=lmax[i], label=llabel[i],
                     resolution=0.02, orient=tk.HORIZONTAL)
            for i in range(len(llabel))
        ]

        # Bind lens sliders to slow update on button release only and pack
        for i, l in enumerate(self.lens_sliders):
            l.bind("<ButtonRelease-1>", self.update_slow)
            l.set(ldef[i])
            l.pack()

        self.snr_slider = tk.Scale(self.image_sliders, from_=10.0, to=300, label='SNR in Mask',
                                   resolution=1.0, orient=tk.HORIZONTAL, command=self.update_plots)
        self.snr_slider.set(self.p.snr)
        self.snr_slider.pack()

        self.mask_slider = tk.Scale(self.image_sliders, from_=0.0, to=1.0, label='Mask Alpha',
                                    resolution=0.1, orient=tk.HORIZONTAL, command=self.update_plots)
        self.mask_slider.set(0.2)
        self.mask_slider.pack()

        self.pix = pixels(self.p.pwd, self.p.wid, 0.0, self.p.n)
        self.src = sersic(self.pix, self.p.srcx, self.p.srcy, self.p.srcr, self.p.srcb, self.p.srcm)
        self.alpha = tessore(self.pix, self.p.gamm, self.p.axro, self.p.mass, self.p.posa)
        self.img = sersic(self.alpha, self.p.srcx, self.p.srcy, self.p.srcr, self.p.srcb, self.p.srcm)

        self.fig = Figure(figsize=(6, 6))

        self.ax1 = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.0, top=1.0, left=0.0, right=1.0)
        self.n_level, self.mask = self.snr_set()
        self.img_noise = self.img + np.random.normal(0.0, self.n_level, self.img.shape)

        self.img_plot = self.ax1.imshow(self.img_noise, extent=[-2, 2, -2, 2],
                                   origin='lower', interpolation='none')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.show()
        self.canvas_w = self.canvas.get_tk_widget()
        self.canvas_w.grid(column=0, row=0)
        self.canvas_w.pack(fill='both', expand=True)

        self.update_slow()

        frame.config(bg='grey')
        frame.pack()

    def update_plots(self, event=None):

        self.n_level, self.mask = self.snr_set()
        self.img_noise = self.img + np.random.normal(0.0, self.n_level, self.img.shape)
        self.caustic = caustic(1.0 - self.lens_sliders[1].get(), self.lens_sliders[2].get(),
                               a=self.lens_sliders[3].get())

        self.ax1.clear()
        self.img_plot = self.ax1.imshow(self.img_noise, extent=[-2, 2, -2, 2],
                                        origin='lower', interpolation='none', cmap=get_cmap('magma'))
        self.mask_plot = self.ax1.contour(self.pix[0], self.pix[1], self.mask,
                                          levels=[0.0], colors='w', alpha=self.mask_slider.get())
        self.source_plot = self.ax1.plot(self.source_sliders[0].get(),
                                         self.source_sliders[1].get(),
                                         'wx')
        self.source_circle = circle_coords(self.source_sliders[2].get(),
                                           self.source_sliders[0].get(),
                                           self.source_sliders[1].get())
        self.source_circle_plot = self.ax1.plot(self.source_circle[0], self.source_circle[1], 'w')
        self.caustic_plot = self.ax1.plot(self.caustic[0], self.caustic[1], 'w', alpha=0.5, lw=0.5)

        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax1.axhline(0.0, color='w', linestyle='-', alpha=0.5, lw=0.5)
        self.ax1.axvline(0.0, color='w', linestyle='-', alpha=0.5, lw=0.5)
        self.canvas.draw()

    def update_fast(self, event=None):

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

        self.src = sersic(self.pix,
                          self.source_sliders[0].get(),
                          self.source_sliders[1].get(),
                          self.source_sliders[2].get(), self.p.srcb,
                          self.source_sliders[3].get())

        self.alpha = tessore(self.pix,
                             self.lens_sliders[0].get(),
                             1.0 - self.lens_sliders[1].get(),
                             self.lens_sliders[2].get(),
                             180.0 * self.lens_sliders[3].get() / np.pi)

        self.img = sersic(self.alpha,
                          self.source_sliders[0].get(),
                          self.source_sliders[1].get(),
                          self.source_sliders[2].get(), self.p.srcb,
                          self.source_sliders[3].get())

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
