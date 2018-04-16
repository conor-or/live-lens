import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QWidget, QDesktopWidget, QApplication,
                             QGridLayout, QMainWindow, QSizePolicy)


class LiveLens(QMainWindow):

    def __init__(self):

        super().__init__()
        self.main_widget = QWidget(self)
        self.init_ui()

    def init_ui(self):

        self.resize(800, 400)
        self.center()

        self.setWindowTitle('Live Lens')

        grid = QGridLayout()
        self.main_widget.setLayout(grid)
        sc = ImageCanvas(self.main_widget, width=5, height=5, dpi=100)
        grid.addWidget(sc, 0, 0)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.show()

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class ImageCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):

        img = np.random.normal(size=(100, 100))
        self.axes.imshow(img, extent=[-2, 2, -2, 2], interpolation='none', origin='lower')

if __name__ == '__main__':

    app = QApplication(sys.argv)
    livelens = LiveLens()
    sys.exit(app.exec_())
