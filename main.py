import Tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tools.lensTools import pixels
from tools.physTools import sersic, tessore
from imp import load_source


class App:

    def __init__(self, master):

        frame = tk.Frame(master)
        # Create 2 buttons

        # Load default parameters
        self.p = load_source('', './templates/default.template').params()

        self.pix = pixels(self.p.pwd, self.p.wid, 0.0, self.p.n)
        self.src = sersic(self.pix, self.p.srcx, self.p.srcy, self.p.srcr, self.p.srcb, self.p.srcm)
        self.alpha = tessore(self.pix, self.p.gamm, self.p.axro, self.p.mass, self.p.posa)
        self.img = sersic(self.alpha, self.p.srcx, self.p.srcy, self.p.srcr, self.p.srcb, self.p.srcm)

        self.slider_srcx = tk.Scale(frame, from_=-0.5, to=0.5, label='Source X',
                                    orient=tk.HORIZONTAL, resolution=0.02, command=self.update_half)
        self.slider_srcy = tk.Scale(frame, from_=-0.5, to=0.5, label='Source Y',
                                    orient=tk.HORIZONTAL, resolution=0.02, command=self.update_half)
        self.slider_srcr = tk.Scale(frame, from_=0.02, to=0.30, label='Source Eff. Rad.',
                                    orient=tk.HORIZONTAL, resolution=0.02, command=self.update_half)
        self.slider_srcm = tk.Scale(frame, from_=0.5, to=5.0, label='Source Sersic Index',
                                    orient=tk.HORIZONTAL, resolution=0.02, command=self.update_half)

        self.slider_axro = tk.Scale(frame, from_=0.0, to=1.0, label='Ellipticity',
                                    orient=tk.HORIZONTAL, resolution=0.02)

        self.slider_axro.bind("<ButtonRelease-1>", self.update_full)

        self.slider_srcx.pack()
        self.slider_srcy.pack()
        self.slider_srcr.pack()
        self.slider_srcm.pack()

        self.slider_axro.pack()

        fig = Figure(figsize=(20, 10))

        ax1 = fig.add_subplot(121)
        ax1.set_xticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        ax1.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.src_plot = ax1.imshow(self.src, extent=[-2, 2, -2, 2],
                                   origin='lower', interpolation='none')

        self.caustic = caustic(self.p.axro, self.p.mass)
        self.caustic_plot, = ax1.plot(self.caustic[0], self.caustic[1], '--')

        ax2 = fig.add_subplot(122)
        ax2.set_xticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        ax2.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.img_plot = ax2.imshow(self.img, extent=[-2, 2, -2, 2],
                                   origin='lower', interpolation='none')

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        frame.pack()

    def update_half(self, event=None):

        self.src = sersic(self.pix,
                          self.slider_srcx.get(),
                          self.slider_srcy.get(),
                          self.slider_srcr.get(), self.p.srcb,
                          self.slider_srcm.get())

        self.img = sersic(self.alpha,
                          self.slider_srcx.get(),
                          self.slider_srcy.get(),
                          self.slider_srcr.get(), self.p.srcb,
                          self.slider_srcm.get())

        self.src_plot.set_data(self.src)
        self.img_plot.set_data(self.img)

        self.canvas.draw()

    def update_full(self, event=None):

        self.src = sersic(self.pix,
                          self.slider_srcx.get(),
                          self.slider_srcy.get(),
                          self.slider_srcr.get(), self.p.srcb,
                          self.slider_srcm.get())

        self.alpha = tessore(self.pix, self.p.gamm, 1.0 - self.slider_axro.get(),
                             self.p.mass, self.p.posa)

        self.img = sersic(self.alpha,
                          self.slider_srcx.get(),
                          self.slider_srcy.get(),
                          self.slider_srcr.get(), self.p.srcb,
                          self.slider_srcm.get())

        ca = caustic(1.0 - self.slider_axro.get(), self.p.mass)

        self.src_plot.set_data(self.src)
        self.img_plot.set_data(self.img)
        self.caustic_plot.set_data(ca)
        self.canvas.draw()


def caustic(f, b, a=0.0):

    theta = np.linspace(0.0, 2 * np.pi, 1000)
    delta = np.hypot(np.cos(theta), np.sin(theta) * f)

    f_ = np.sqrt(1.0 - f ** 2)

    y1 = b * (np.cos(theta) / delta - np.arcsinh(f_ * np.cos(theta) / f) / f_)
    y2 = b * (np.sin(theta) / delta - np.arcsin(f_ * np.sin(theta)) / f_)

    y1_ = y1 * np.cos(a) - y2 * np.sin(a)
    y2_ = y1 * np.sin(a) + y2 * np.cos(a)

    return y1_, y2_



root = tk.Tk()
app = App(root)
