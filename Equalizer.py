import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QFileDialog
from gui import Ui_MainWindow
import os
from scipy.io import wavfile
import numpy as np
import pyqtgraph as pg
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl


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

        self.path = None

        self.media_player_status = 0
        self.current_position = 0


        self.data_ranges = [None] * 10

        self.sliders = [
            self.gui.slider1, self.gui.slider2, self.gui.slider3, self.gui.slider4, self.gui.slider5,
            self.gui.slider6, self.gui.slider7, self.gui.slider8, self.gui.slider9, self.gui.slider10
        ]
        for i in range(10):
            self.connect_sliders(i)
        self.gui.actionOpen.triggered.connect(self.open_wav_file)
        self.gui.actionSave.triggered.connect(self.save_wav_file)

        
        # Create a QMediaPlayer instance for playing audio
        self.media_player = QMediaPlayer()
        self.media_player.stateChanged.connect(self.on_media_state_changed)

        # Connect the button click event to the play_file method
        self.gui.btn_play.clicked.connect(self.play_file)

        self.gui.btn_rewind.clicked.connect(self.restart_file)

        self.gui.btn_pan_left.clicked.connect(self.seek_backward)
        self.gui.btn_pan_right.clicked.connect(self.seek_forward)

    
    def seek_forward(self):
        # Get the current position in milliseconds
        current_position = self.media_player.position()

        # Seek forward by 5 seconds (5000 milliseconds)
        new_position = current_position + 5000

        # Set the new position
        self.media_player.setPosition(new_position)
    
    def seek_backward(self):
        # Get the current position in milliseconds
        current_position = self.media_player.position()

        # Seek forward by 5 seconds (5000 milliseconds)
        new_position = current_position - 5000

        # Set the new position
        self.media_player.setPosition(new_position)


    def restart_file(self):
        if self.sample_rate is not None:
            # Create a QMediaContent object with the WAV file path
            media_content = QMediaContent(QUrl.fromLocalFile(self.path))

            # Set the media content to the media player
            self.media_player.setMedia(media_content)

            self.media_player.stop()
            self.media_player.play()


    def play_file(self):
        if self.sample_rate is not None:
            # Create a QMediaContent object with the WAV file path
            media_content = QMediaContent(QUrl.fromLocalFile(self.path))

            # Set the media content to the media player
            self.media_player.setMedia(media_content)

            if self.media_player_status == 1:
                # Pause the audio
                self.current_position = self.media_player.position()
                self.media_player.pause()
                self.media_player_status = 0
            else:
                # Set the position to the stored value
                self.media_player.setPosition(self.current_position)
                self.media_player.play()
                self.media_player_status = 1
            

    def on_media_state_changed(self, state):
        # Handle media player state changes, e.g., update UI based on playback status
        if state == QMediaPlayer.PlayingState:
            print("Audio is playing")
        elif state == QMediaPlayer.StoppedState:
            print("Audio playback stopped")
        elif state == QMediaPlayer.PausedState:
            print("Audio playback paused")


    def save_wav_file(self):
        wavfile.write(f'output.wav', self.sample_rate, self.data_modified)


    def open_wav_file(self):
        try:
            files_name = QFileDialog.getOpenFileName(self, 'Open WAV File', os.getenv('HOME'), "WAV files (*.wav)")
            self.path = files_name[0]
            if self.path:
                sample_rate, signal = wavfile.read(self.path)
                self.data = signal
                self.sample_rate = sample_rate
                self.data_fft = np.fft.fft(signal)

                self.data_modified = self.data    
                self.data_modified_fft = self.data_fft


                frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
                self.section_width = len(frequencies) // 10
                for i in range(10):
                    start_idx = i * self.section_width
                    end_idx = (i + 1) * self.section_width
                    self.data_ranges[i] = [start_idx, end_idx]
                self.plot_on_main(self.data, self.data_fft)
                self.plot_on_secondary(self.data_modified, self.data_modified_fft)

        except Exception as e:
            print(f"Error: {e}")


    def plot_on_main(self, data, freq):
        self.gui.plot_input_sig_time.clear()
        self.gui.plot_input_sig_freq.clear()

        self.gui.plot_input_sig_time.plot(np.linalg.norm(data, axis=1), pen="r")
        self.gui.plot_input_sig_freq.plot(np.linalg.norm(freq, axis=1), pen="r")


    def plot_on_secondary(self, data, freq):
        self.gui.plot_output_sig_time.clear()
        self.gui.plot_output_sig_freq.clear()

        self.gui.plot_output_sig_time.plot(np.linalg.norm(data, axis=1), pen="r")
        self.gui.plot_output_sig_freq.plot(np.linalg.norm(freq, axis=1), pen="r")

    def set_bands_gains_sliders(self):
        for i in range(10):
            self.sliders[i].setMinimum(1)
            self.sliders[i].setMaximum(2)
            self.sliders[i].setValue(1)
            self.sliders[i].setTickInterval(0.1)

    def connect_sliders(self, index):
        self.sliders[index].valueChanged.connect(lambda: self.mult_freqs(index))

    def mult_freqs(self, index):
        self.data_modified_fft = self.multiply_fft(
            self.data_modified_fft,
            self.data_ranges[index][0],
            self.data_ranges[index][1],
            10**((self.sliders[index].value()) / 20),
            std_gaussian=self.section_width / 100,
            mult_window=self.mult_window
        )

        self.data_modified = np.fft.ifft(self.data_modified_fft)
        self.data_modified = self.data_modified.real.astype(np.int16)  # Real part only

        # wavfile.write(f'output.wav', self.sample_rate, self.data_modified)
        self.plot_on_secondary(self.data_modified, self.data_modified_fft)

    def multiply_fft(self, data, start, end, index, std_gaussian, mult_window):
        modified_data = data.copy()

        if mult_window == "rectangle":
            modified_data[start:end] = self.data_fft[start:end] * index

        elif mult_window == "hamming":
            hamming_window = np.hamming(end - start) * index
            modified_data[start:end] = self.data_fft[start:end] * hamming_window

        elif mult_window == "hanning":
            hanning_window = np.hanning(end - start) * index
            modified_data[start:end] = self.data_fft[start:end] * hanning_window

        elif mult_window == "gaussian":
            gaussian_window = np.exp(-0.5 * ((np.arange(end - start) - (end - start) / 2) / std_gaussian) ** 2) * index
            modified_data[start:end] = self.data_fft[start:end] * gaussian_window

        return modified_data


# Run Application
app = QApplication(sys.argv)
win = Equalizer()
win.show()
app.exec()
