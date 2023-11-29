"""Multi-use Equalizer that can sample many types of signals such as: Musical instruments - Medical Signals (e.g.: ECG) - Animal Sounds - Audio Samples"""
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow
from gui import Ui_MainWindow
from cProfile import label
from matplotlib.pyplot import xlabel
from gui import Ui_MainWindow
import os
import math
from scipy.fftpack import fft, fftfreq
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import pyqtgraph as pg

class EqualizerGUI(Ui_MainWindow):
    def setupUi(self, MainWindow):
        Ui_MainWindow.setupUi(self, MainWindow)

class Equalizer(QMainWindow):
    def __init__(self):
        super(Equalizer, self).__init__()
        self.gui = EqualizerGUI()
        self.gui.setupUi(self)

        self.data = []
        self.data_fft = None

        self.data_modified = []
        self.data_modified_fft = None

        self.sample_rate = None
        self.data_ranges = []

        self.mult_window = "rectangle"
        self.section_width = None

        self.data_ranges = [None] * 10

        self.sliders = [
            self.gui.slider1, self.gui.slider2, self.gui.slider3, self.gui.slider4, self.gui.slider5,
            self.gui.slider6, self.gui.slider7, self.gui.slider8, self.gui.slider9, self.gui.slider10
        ]
        for i in range(10):
            self.connect_sliders(i)
        self.gui.actionOpen.triggered.connect(self.open_wav_file)



        
    def open_wav_file(self):
        try:
            files_name = QFileDialog.getOpenFileName(self, 'Open WAV File', os.getenv('HOME'), "WAV files (*.wav)")
            path = files_name[0]
            if path:
                sample_rate, signal = wavfile.read(path)
                # Store the data for later use
                self.data = signal
                self.sample_rate = sample_rate
                self.data_fft = np.fft.fft(signal)

                frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
                self.section_width = len(frequencies) // 10
                for i in range(10):
                    start_idx = i * self.section_width
                    end_idx = (i + 1) * self.section_width
                    self.data_ranges[i] = [start_idx, end_idx]
                
                self.plotOnMain(self.data, self.data_fft)
                self.set_bands_gains_sliders()
        except Exception as e:
            print(f"Error: {e}")
    


    
    def plotOnMain(self, data, freq):
        time = np.arange(0, len(data)) / self.sample_rate
        time = list(time)
        data = list(data)
        # self.gui.plot_input_sig_time.plot(time ,data, pen="r")
        print(data, " \n", time) 
        self.gui.plot_input_sig_freq.plot(np.abs(freq), pen="r")


    def plotOnSecondary(self):
        self.gui.plot_output_sig_time.plot(self.data_modified.real, pen="r")
        self.gui.plot_output_sig_freq.plot(np.abs(self.data_modified_fft), pen="r")

    
    def set_bands_gains_sliders(self):
        for i in range(10):
            self.sliders[i].setMinimum(1)
            self.sliders[i].setMaximum(2)
            self.sliders[i].setValue(1)
            self.sliders[i].setTickInterval(0.1)
    

    def connect_sliders(self, index):
        self.sliders[index].valueChanged.connect(lambda: self.mult_freqs(index))

    #TODO - Make it use a dialog
    def save_wav_file(self):
        try:
            wavfile.write(f'output.wav', self.sample_rate, self.data_modified)
        except:
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setWindowTitle("Save File")
            msg.setText("Sound file saved successfully!")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok) 
            msg.exec()
            return
        
    
    def mult_freqs(self, index):
        self.data_modified_fft = self.multiply_fft(self.data, self.data[index][0], self.data[index][1], self.sliders[index].value(), std_gaussian= self.section_width/100, mult_window = self.mult_window)

        self.data_modified = np.fft.ifft(self.data_modified_fft)
        # Convert the signal to the appropriate data type for saving as a WAV file
        self.data_modified = self.data_modified.real.astype(np.int16)  # Real part only





    def multiply_fft(self, data, start, end, index, std_gaussian, mult_window):
        # Create a copy of the data to avoid modifying the original array
        modified_data = data.copy()
        
        if mult_window == "rectangle":
            modified_data[start:end] *= index
        
        elif mult_window == "hamming":
            hamming_window = np.hamming(end - start) * index
            modified_data[start:end, 0] = data[start:end, 0] * hamming_window
            modified_data[start:end, 1] = data[start:end, 1] * hamming_window
        
        elif mult_window == "hanning":
            hanning_window = np.hanning(end - start) * index
            modified_data[start:end, 0] = data[start:end, 0] * hanning_window
            modified_data[start:end, 1] = data[start:end, 1] * hanning_window

        elif mult_window == "gaussian":
            gaussian_window = np.exp(-0.5 * ((np.arange(end - start) - (end - start) / 2) / std_gaussian) ** 2) * index
            modified_data[start:end, 0] = data[start:end, 0] * gaussian_window
            modified_data[start:end, 1] = data[start:end, 1] * gaussian_window
        
        return modified_data
        
# Run Application
app = QApplication(sys.argv)
win = Equalizer()
win.show()
app.exec()