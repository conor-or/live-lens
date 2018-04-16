import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QToolTip, QPushButton,
                             QDesktopWidget, QMainWindow, QAction, qApp,
                             QTextEdit, QHBoxLayout, QVBoxLayout, QGridLayout,
                             QLCDNumber, QSlider, QSizePolicy)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class LiveLens(QMainWindow):

    def __init__(self):

        super().__init__()

        self.init_ui()

    def init_ui(self):

        # Set up the central widget and grid layout
        main = QWidget(self)
        self.setCentralWidget(main)
        layout = QGridLayout()
        main.setLayout(layout)

        # Set font for tool tips
        QToolTip.setFont(QFont('SansSerif', 10))

        img = PlotCanvas(self)
        source_sliders = QWidget(self)
        source_sliders_layout = QVBoxLayout()

        for i in range(5):
            slider = QSlider(Qt.Horizontal, source_sliders)
            source_sliders_layout.addWidget(slider)

        layout.addWidget(img, 0, 0)
        layout.addWidget(source_sliders, 0, 1)

        # self.resize()
        self.center()
        self.setWindowTitle('Live Lens')

        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=5, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        # FigureCanvas.setSizePolicy(self,
        #                            QSizePolicy.Expanding,
        #                            QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):

        data = np.random.normal(size=(100, 100))
        ax = self.figure.add_subplot(111)
        ax.imshow(data, extent=[-2, 2, -2, 2], interpolation='none',
                  cmap=plt.get_cmap('plasma'))
        self.draw()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    livelens = LiveLens()
    sys.exit(app.exec_())
