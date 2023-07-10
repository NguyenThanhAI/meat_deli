# import pyconcrete

import sys

from PyQt5 import QtWidgets
from main import MainWindow


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.start()
    widget.show()
    sys.exit(app.exec_())
