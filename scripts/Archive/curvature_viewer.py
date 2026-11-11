import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys

def view_curvature_pyqtgraph(curvature, dt):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    curvature = np.asarray(curvature)

    if curvature.shape[0] < curvature.shape[1]:
        n_frames = curvature.shape[1]
        get_frame = lambda i: curvature[:, i]
        N = curvature.shape[0]
    else:
        n_frames = curvature.shape[0]
        get_frame = lambda i: curvature[i]
        N = curvature.shape[1]

    s = np.linspace(0, 1, N)

    win = pg.GraphicsLayoutWidget(show=True)
    plot = win.addPlot()
    plot.showGrid(x=False, y=False)

    curve = plot.plot([], [], pen=pg.mkPen('w', width=2))

    frame = 0
    interval_ms = int(dt * 1000)
    paused = False

    def update():
        nonlocal frame
        y = get_frame(frame)
        curve.setData(s, y)
        frame = (frame + 1) % n_frames

    def step_frame():
        y = get_frame(frame)
        curve.setData(s, y)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(interval_ms)

    def keyPressEvent(event):
        nonlocal frame, paused, interval_ms
        if event.key() == QtCore.Qt.Key_Space:
            paused = not paused
            timer.stop() if paused else timer.start(interval_ms)
        elif event.key() == QtCore.Qt.Key_Right:
            frame = (frame + 1) % n_frames
            step_frame()
        elif event.key() == QtCore.Qt.Key_Left:
            frame = (frame - 1) % n_frames
            step_frame()
        elif event.key() == QtCore.Qt.Key_Plus:
            interval_ms = max(5, interval_ms - 5)
            timer.start(interval_ms)
        elif event.key() == QtCore.Qt.Key_Minus:
            interval_ms += 5
            timer.start(interval_ms)

    win.keyPressEvent = keyPressEvent

    sys.exit(app.exec_())
