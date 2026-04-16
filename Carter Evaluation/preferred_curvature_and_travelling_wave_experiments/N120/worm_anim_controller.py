import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys

def view_worm_pyqtgraph(worm_positions, dt):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])


    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle("Simple Worm Viewer (pyqtgraph)")
    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.showGrid(x=True, y=True)

    worm = plot.plot([], [], pen=pg.mkPen('w', width=2))

    n_frames = len(worm_positions)
    frame = 0
    interval_ms = int(dt * 1000)
    paused = False

    def update():
        nonlocal frame
        x = worm_positions[frame]
        worm.setData(x[:, 0], x[:, 2])
        frame = (frame + 1) % n_frames

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
            update()
        elif event.key() == QtCore.Qt.Key_Left:
            frame = (frame - 1) % n_frames
            update()
        elif event.key() == QtCore.Qt.Key_Plus:
            interval_ms = max(5, interval_ms - 5)
            timer.start(interval_ms)
        elif event.key() == QtCore.Qt.Key_Minus:
            interval_ms += 5
            timer.start(interval_ms)

    win.keyPressEvent = keyPressEvent

    print("""
Controls:
  space  -> play / pause
  L / R  -> step frames
  + / -  -> speed up / slow down
  mouse  -> zoom / pan
""")

    sys.exit(app.exec_())
