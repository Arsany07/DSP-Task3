"""Multi-use Equalizer that can sample many types of signals such as: Musical instruments - Medical Signals (e.g.: ECG) - Animal Sounds - Audio Samples"""
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from gui import Ui_MainWindow

class Equalizer(QMainWindow):
    def __init__(self):
        super(QMainWindow).__init__()
        self.setupUi(self)
        
        
app = QApplication(sys.argv)
win = Equalizer() # Change to class name
win.show()
app.exec()