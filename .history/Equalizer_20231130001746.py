import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QFileDialog
from gui import Ui_MainWindow
import os
from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from pydub import AudioSegment
import matplotlib.pyplot as plt


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
        self.frequencies = None
        self.data_modified_frequencies = None
        self.sample_rate = None
        self.data_ranges = []
        self.mult_window = "rectangle"
        self.section_width = None

        self.path = None

        self.std = 10

        self.current_position = 0
        self.media_player_status = 0

        self.data_ranges = [None] * 10

        self.sliders = [
            self.gui.slider1, self.gui.slider2, self.gui.slider3, self.gui.slider4, self.gui.slider5,
            self.gui.slider6, self.gui.slider7, self.gui.slider8, self.gui.slider9, self.gui.slider10
        ]
        
        self.slider_wgts = [ self.gui.wgt_sld_1, self.gui.wgt_sld_2, self.gui.wgt_sld_3, self.gui.wgt_sld_4,
                            self.gui.wgt_sld_5, self.gui.wgt_sld_6, self.gui.wgt_sld_7, self.gui.wgt_sld_8,
                            self.gui.wgt_sld_9, self.gui.wgt_sld_10]
        
        self.views = [self.gui.plot_input_sig_freq, self.gui.plot_input_sig_time, self.gui.plot_input_sig_spect, self.gui.plot_output_sig_freq,
                      self.gui.plot_output_sig_time, self.gui.plot_output_sig_spect]
        for i in range(10):
            self.connect_sliders(i)
        self.gui.actionOpen.triggered.connect(self.open_wav_file)
        self.gui.actionSave.triggered.connect(self.save_wav_file)


        # Create a QMediaPlayer instance for playing audio
        self.media_player = QMediaPlayer()
        self.media_player.stateChanged.connect(self.on_media_state_changed)
        
        
        # Connect the button click event to the play_file method
        self.gui.btn_play_input.clicked.connect(lambda: self.play_file(self.media_player, path=self.path))
        self.gui.btn_rewind_input.clicked.connect(lambda: self.restart_file(self.media_player, path=self.path))
        self.gui.btn_pan_left_input.clicked.connect(lambda: self.seek_backward(self.media_player))
        self.gui.btn_pan_right_input.clicked.connect(lambda: self.seek_forward(self.media_player))


        self.media_player_output = QMediaPlayer()
        self.media_player_output.stateChanged.connect(self.on_media_state_changed_output)
        
        
        # Connect the button click event to the play_file method
        self.gui.btn_play_output.clicked.connect(lambda: self.play_file(self.media_player_output, path="output.wav"))
        self.gui.btn_rewind_output.clicked.connect(lambda: self.restart_file(self.media_player_output, path="output.wav"))
        self.gui.btn_pan_left_output.clicked.connect(lambda: self.seek_backward(self.media_player_output))
        self.gui.btn_pan_right_output.clicked.connect(lambda: self.seek_forward(self.media_player_output))

        # Connect checkboxes to show/hide spectrograms
        self.gui.chkbx_spect_input.stateChanged.connect(self.hide_input_spectrogram)
        self.gui.chkbx_spect_output.stateChanged.connect(self.hide_output_spectrogram)
            
        # Connect Combo boxes
        self.gui.cmbx_mode_selection.currentIndexChanged.connect(self.switch_modes)
        self.gui.cmbx_multWindow.currentIndexChanged.connect(self.update_window)


        self.gui.slider_amplitude_2.valueChanged.connect(self.set_std)
        self.gui.slider_amplitude_2.setEnabled(False)
        
        # Window setup at first launch
        self.hide_input_spectrogram()
        self.hide_output_spectrogram()
        self.link_views()
        self.apply_optimizations_to_views()

    #=============================== Function Definitions ===============================#
    
    
    # Links Views
    def link_views(self):
        self.gui.plot_input_sig_time.setXLink(self.gui.plot_output_sig_time)
        self.gui.plot_input_sig_time.setYLink(self.gui.plot_output_sig_time)
        self.gui.plot_input_sig_spect.setXLink(self.gui.plot_output_sig_spect)
        self.gui.plot_input_sig_spect.setYLink(self.gui.plot_output_sig_spect)
        self.gui.plot_input_sig_freq.setXLink(self.gui.plot_output_sig_freq)
        self.gui.plot_input_sig_freq.setYLink(self.gui.plot_output_sig_freq)

    
    def apply_optimizations_to_views(self):
        for view in self.views:
            view.getPlotItem.setDownsampling(auto=False, ds = 2, mode = 'mean')
            view.getPlotItem.setClipToView(True)

        
    #TODO - CHANGE INTO ONE FUNCTION TO AVOID REPITITION

    # "Mode Changing" methods
    def change_mode_uniform(self):
        
        print ("Uniform mode")
        
        for slider in self.slider_wgts[4:10]:
            slider.setVisible(True)
        for i, widget in enumerate(self.slider_wgts):
            widget.findChild(QtWidgets.QLabel).setText(f"Slider {i+1}")
            
    def change_mode_instruments(self):
        
        print ("Instrument Mode")
        
        for slider in self.slider_wgts[4:10]:
            slider.setVisible(False)
        for i, widget in enumerate(self.slider_wgts):
            widget.findChild(QtWidgets.QLabel).setText(f"Instrument {i+1}")
        # for i, label in enumerate(self.gui.slider_wgts.findChildren(QtWidgets.QLabel)):
        #     label.setText(f"Instrument {i}")

                
        
    def change_mode_animals(self):
        
        print ("Animal Mode")

        for slider in self.slider_wgts[4:10]:
            slider.setVisible(False)
            
        for i, widget in enumerate(self.slider_wgts):
            widget.findChild(QtWidgets.QLabel).setText(f"Animal {i+1}")
        
        # for slider in self.sliders[4:9]:
        #     slider.setVisible(False)
        # for i, label in enumerate(self.gui.wgt_sliders.findChildren(QtWidgets.QLabel)):
        #     label.setText(f"animal {i}")
    
    def change_mode_ECG(self):
        print ("ECG Mode")
        for slider in self.slider_wgts[3:10]:
            slider.setVisible(False)
            
        for i, widget in enumerate(self.slider_wgts):
            widget.findChild(QtWidgets.QLabel).setText(f"Arrythmia {i+1}")
        pass
    
    def switch_modes(self):
        mode = self.gui.cmbx_mode_selection.currentText()
        print(mode)
        
        match mode:
            
            case "Uniform Range Mode":
                self.change_mode_uniform()
                
            case "Musical Instruments Mode":
                self.change_mode_instruments()
                
            case "Animal Sounds Mode":
                self.change_mode_animals()
                
            case "ECG Abnormalities Mode":
                self.change_mode_ECG()
                
            case _:
                print ("Default Case")
            

    
    

    def set_std(self):
        self.std = self.gui.slider_amplitude_2.value()
        self.gui.lbl_value_amp_3.setText(str(self.gui.slider_amplitude_2.value()))
        
    def hide_input_spectrogram(self):
        self.gui.plot_input_sig_spect.setVisible(self.gui.chkbx_spect_input.isChecked())

    def hide_output_spectrogram(self):
        self.gui.plot_output_sig_spect.setVisible(self.gui.chkbx_spect_output.isChecked())

    def update_window(self, index):
        # Get the selected item from the combo box
        selected_item = self.gui.cmbx_multWindow.currentText()
        self.gui.slider_amplitude_2.setEnabled(False)

        if selected_item == "Rectangle":
            self.mult_window = "rectangle"

        elif selected_item == "Hamming":
            self.mult_window = "hamming"

        elif selected_item == "Hanning":
            self.mult_window = "hanning"

        elif selected_item == "Gaussian":
            self.mult_window = "gaussian"
            self.gui.slider_amplitude_2.setEnabled(True)

        print(f"Selected window: {self.mult_window}")


    
    def seek_forward(self, media):
        current_position = media.position()
        new_position = current_position + 5000
        media.setPosition(new_position)
    
    def seek_backward(self, media):
        current_position = media.position()
        new_position = current_position - 5000
        media.setPosition(new_position)


    def restart_file(self, media, path):
        if self.sample_rate is not None:
            media_content = QMediaContent(QUrl.fromLocalFile(path))
            media.setMedia(media_content)
            media.play()


    def play_file(self, media, path):
        if self.sample_rate is not None:
            media_content = QMediaContent(QUrl.fromLocalFile(path))
            media.setMedia(media_content)

            if self.media_player_status == 1:
                self.current_position = media.position()
                media.pause()
                self.media_player_status = 0
            else:              
                media.play()
                media.setPosition(self.current_position)
                self.media_player_status = 1

            

    def on_media_state_changed(self, state):
        # Handle media player state changes, e.g., update UI based on playback status
        if state == QMediaPlayer.PlayingState:
            print("Audio is playing")
        elif state == QMediaPlayer.StoppedState:
            print("Audio playback stopped")
        elif state == QMediaPlayer.PausedState:
            print("Audio playback paused")
    
    def on_media_state_changed_output(self, state):
        # Handle media player state changes, e.g., update UI based on playback status
        if state == QMediaPlayer.PlayingState:
            print("Output Audio is playing")
        elif state == QMediaPlayer.StoppedState:
            print("Output Audio playback stopped")
        elif state == QMediaPlayer.PausedState:
            print("Output Audio playback paused")


    def save_wav_file(self):
        normalized_array = np.interp( self.data_modified, (np.min(self.data_modified), np.max(self.data_modified)), (-32768, 32767)).astype(np.int16)
        wavfile.write(f'output.wav', self.sample_rate, normalized_array)


    def open_wav_file(self):
        try:
            files_name = QFileDialog.getOpenFileName(self, 'Open WAV File', os.getenv('HOME'), "WAV files (*.wav)")
            self.path = files_name[0]
            if self.path:
                sample_rate, signal = wavfile.read(self.path)
                self.data = signal
                self.sample_rate = sample_rate
                self.data_fft = np.fft.fft(signal)
                self.frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)

                self.data_modified = self.data    
                self.data_modified_fft = self.data_fft
                self.data_modified_frequencies = self.frequencies

                self.section_width = len(self.frequencies) // 10
                for i in range(10):
                    start_idx = i * self.section_width
                    end_idx = (i + 1) * self.section_width
                    self.data_ranges[i] = [start_idx, end_idx]
                self.plot_on_main(self.data, self.frequencies)
                self.plot_on_secondary(self.data_modified, self.data_modified_frequencies)

                self.plot_spectrogram_main()
                self.plot_spectrogram_secondary()

        except Exception as e:
            print(f"Error: {e}")


    def plot_spectrogram(self, ax, data, sample_rate, title):
        # Compute Spectrogram
        f, t, sxx = spectrogram(data, fs=sample_rate)

        # Plot Spectrogram
        img = pg.ImageItem()
        img.setImage(np.log(sxx + 1))
        ax.addItem(img)

        # Set labels and colormap
        ax.setLabel('left', 'Frequency', units='Hz')
        ax.setLabel('bottom', 'Time', units='s')
        colormap = pg.colormap.get('viridis')
        img.setColorMap(colormap)
        ax.setTitle(title)



    def plot_spectrogram_main(self):
        self.plot_spectrogram(self.gui.plot_input_sig_spect, self.data, self.sample_rate, "Input Spectrogram")

    def plot_spectrogram_secondary(self):
        self.plot_spectrogram(self.gui.plot_output_sig_spect, self.data_modified, self.sample_rate, "Output Spectrogram")






    def plot_on_main(self, data, freq):
        self.gui.plot_input_sig_time.clear()
        self.gui.plot_input_sig_freq.clear()

        self.gui.plot_input_sig_time.plot(np.linalg.norm(data, axis=1), pen="r")
        self.gui.plot_input_sig_freq.plot(np.abs(freq), pen="r")
        print(np.abs(freq))

    def plot_on_secondary(self, data, freq):
        self.gui.plot_output_sig_time.clear()
        self.gui.plot_output_sig_freq.clear()

        self.gui.plot_output_sig_time.plot(np.linalg.norm(data, axis=1), pen="r")
        self.gui.plot_output_sig_freq.plot(np.abs(freq), pen="r")

    def set_bands_gains_sliders(self):
        for i in range(10):
            self.sliders[i].setMinimum(-30)
            self.sliders[i].setMaximum(30)
            self.sliders[i].setValue(0)
            self.sliders[i].setTickInterval(1)

    def connect_sliders(self, index):
        self.sliders[index].valueChanged.connect(lambda: self.mult_freqs(index))

    def mult_freqs(self, index):
        self.data_modified_frequencies = self.multiply_fft(
            self.data_modified_frequencies,
            self.data_ranges[index][0],
            self.data_ranges[index][1],
            10**(self.sliders[index].value()/20),
            std_gaussian=self.section_width / self.std,
            mult_window=self.mult_window
        )

        self.data_modified = np.fft.ifft(self.data_modified_fft)
        self.data_modified = self.data_modified.real.astype(np.int64)  # Real part only

        
        self.plot_on_secondary(self.data_modified, self.data_modified_frequencies)
        self.plot_spectrogram_secondary()

    def multiply_fft(self, data, start, end, index, std_gaussian, mult_window):
        modified_data = data.copy()

        if mult_window == "rectangle":
            modified_data[start:end] = self.frequencies[start:end] * index

        elif mult_window == "hamming":
            hamming_window = np.hamming(end - start)[:, np.newaxis] * index
            modified_data[start:end] = self.frequencies[start:end] * hamming_window

        elif mult_window == "hanning":
            hanning_window = np.hanning(end - start)[:, np.newaxis] * index
            modified_data[start:end] = self.frequencies[start:end] * hanning_window

        elif mult_window == "gaussian":
            gaussian_window = np.exp(-0.5 * ((np.arange(end - start) - (end - start) / 2) / std_gaussian) ** 2)[:, np.newaxis] * index
            modified_data[start:end] = self.frequencies[start:end] * gaussian_window

        return modified_data


# Run Application
app = QApplication(sys.argv)
win = Equalizer()
win.show()
app.exec()
