import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QFileDialog
from gui import Ui_MainWindow
import io
import os
from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import matplotlib.pyplot as plt
import librosa



class EqualizerGUI(Ui_MainWindow):
    def setupUi(self, MainWindow):
        Ui_MainWindow.setupUi(self, MainWindow)

class Equalizer(QMainWindow):
    
    def __init__(self):
        super(Equalizer, self).__init__()
        self.gui = EqualizerGUI()
        self.gui.setupUi(self)
        self.setFocus()


        self.instrument_ranges = [
            [1000, 2000], [1000, 2000], [1000, 2000], [1000, 2000]
        ]

        self.animal_ranges = [
            [1000, 2000], [1000, 2000], [1000, 2000], [1000, 2000]
        ]
        
        self.data = []
        self.data_fft = None
        self.time = []
        self.data_modified = []
        self.data_modified_fft = None
        self.frequencies = None
        self.data_modified_frequencies = None
        self.sample_rate = None
        self.data_ranges = []
        self.mult_window = "rectangle"
        self.section_width = None

        self.path = None

        self.std = 100

        self.current_position = 0
        
        self.speed_state = 1 # Will cycle between 1 and 6 for x1.0, x1.25, x1.5, x1.75, x2.0, x0.5
        # self.playback_speed = 1.0 # The actual value that'll control speed
        
        self.media_player_status = 0

        self.data_ranges = [None] * 10

        self.sliders = [
            self.gui.slider1, self.gui.slider2, self.gui.slider3, self.gui.slider4, self.gui.slider5,
            self.gui.slider6, self.gui.slider7, self.gui.slider8, self.gui.slider9, self.gui.slider10
        ]

        self.sliders_gains = [
            self.gui.lnEdit_gain_slider_1, self.gui.lnEdit_gain_slider_2, self.gui.lnEdit_gain_slider_3, self.gui.lnEdit_gain_slider_4, self.gui.lnEdit_gain_slider_5,
            self.gui.lnEdit_gain_slider_6, self.gui.lnEdit_gain_slider_7, self.gui.lnEdit_gain_slider_8, self.gui.lnEdit_gain_slider_9, self.gui.lnEdit_gain_slider_10
        ]

        self.sliders_freqs = [
            self.gui.lnEdit_freq_slider_1, self.gui.lnEdit_freq_slider_2, self.gui.lnEdit_freq_slider_3, self.gui.lnEdit_freq_slider_4, self.gui.lnEdit_freq_slider_5,
            self.gui.lnEdit_freq_slider_6, self.gui.lnEdit_freq_slider_7, self.gui.lnEdit_freq_slider_8, self.gui.lnEdit_freq_slider_9, self.gui.lnEdit_freq_slider_10
        ]
        
        self.slider_wgts = [ self.gui.wgt_sld_1, self.gui.wgt_sld_2, self.gui.wgt_sld_3, self.gui.wgt_sld_4,
                            self.gui.wgt_sld_5, self.gui.wgt_sld_6, self.gui.wgt_sld_7, self.gui.wgt_sld_8,
                            self.gui.wgt_sld_9, self.gui.wgt_sld_10]
        
        self.views = [self.gui.plot_input_sig_freq, self.gui.plot_input_sig_time, self.gui.plot_input_sig_spect, self.gui.plot_output_sig_freq,
                      self.gui.plot_output_sig_time, self.gui.plot_output_sig_spect]
        for i in range(10):
            self.connect_sliders(i)

        # self.connect_sliders()
        self.gui.actionOpen.triggered.connect(self.open_wav_file)
        self.gui.actionSave.triggered.connect(self.save_wav_file)


        # Create a QMediaPlayer instance for playing audio
        # Input #
        self.media_player_input = QMediaPlayer()
        
        # Vertical line to act as seeker on plot
        
        self.medPlayer_seeker = pg.InfiniteLine(pos = self.media_player_input.position(), angle = 90, pen = pg.mkPen('y'), movable =  True)
        self.media_player_input.stateChanged.connect(self.on_media_state_changed)
        self.media_player_input.positionChanged.connect(lambda position: self.medPlayer_seeker.setValue(position))
        
        # Output #
        self.media_player_output = QMediaPlayer()
        # Vertical line to act as seeker on plot
        
        self.media_player_output.stateChanged.connect(self.on_media_state_changed_output)
        self.media_player_output.positionChanged.connect(lambda position: self.medPlayer_seeker.setValue(position))
        
        # Allow Scrubbing with seeker
        # self.medPlayer_seeker.sigPositionChangeFinished.connect(self.update_player_position)
        
        self.gui.btn_reset_sliders.clicked.connect(self.reset_sliders)
        
        
        # Connect the button click event to the play_file method
        self.gui.btn_play_input.clicked.connect(lambda: self.play_file(self.media_player_input))
        self.gui.btn_rewind_input.clicked.connect(lambda: self.restart_file(self.media_player_input, path=self.path))
        self.gui.btn_pan_left_input.clicked.connect(lambda: self.seek_backward(self.media_player_input))
        self.gui.btn_pan_right_input.clicked.connect(lambda: self.seek_forward(self.media_player_input))


        
        
        # Connect the button click event to the play_file method
        self.gui.btn_play_output.clicked.connect(lambda: self.play_file(self.media_player_output))
        self.gui.btn_rewind_output.clicked.connect(lambda: self.restart_file(self.media_player_output, path="output.wav"))
        self.gui.btn_pan_left_output.clicked.connect(lambda: self.seek_backward(self.media_player_output))
        self.gui.btn_pan_right_output.clicked.connect(lambda: self.seek_forward(self.media_player_output))

        # Connect checkboxes to show/hide spectrograms
        self.gui.chkbx_spect_input.stateChanged.connect(self.hide_input_spectrogram)
        self.gui.chkbx_spect_output.stateChanged.connect(self.hide_output_spectrogram)
            
        # Connect Combo boxes
        self.gui.cmbx_mode_selection.currentIndexChanged.connect(self.switch_modes)
        self.gui.cmbx_multWindow.currentIndexChanged.connect(self.update_window)
        self.plot_multWindow_rectangle()


        self.gui.slider_amplitude_2.valueChanged.connect(self.set_std)
        self.gui.slider_amplitude_2.setEnabled(False)
        
        
        # CONNECT PLOT CONTROL BUTTONS
        # self.gui.btn_pan_left_linked.clicked.connect()
        # self.gui.btn_pan_right_linked.clicked.connect()
        # self.gui.btn_play_linked.clicked.connect()
        # self.gui.btn_zoom_in.clicked.connect()
        # self.gui.btn_zoom_out.clicked.connect()
        self.gui.btn_speed.clicked.connect(self.change_speed)
        
        
        # Window setup at first launch
        self.hide_input_spectrogram()
        self.hide_output_spectrogram()
        self.link_views()
        # self.apply_optimizations_to_views()
        self.gui.wgt_multWindow_amp.setVisible(False)

    #=============================== Function Definitions ===============================#
    
    
    # Links Views
    def link_views(self):
        # self.gui.plot_input_sig_time.setXLink(self.gui.plot_output_sig_time)
        # self.gui.plot_input_sig_time.setYLink(self.gui.plot_output_sig_time)
        # self.gui.plot_input_sig_spect.setXLink(self.gui.plot_output_sig_spect)
        # self.gui.plot_input_sig_spect.setYLink(self.gui.plot_output_sig_spect)
        # self.gui.plot_input_sig_freq.setXLink(self.gui.plot_output_sig_freq)
        # self.gui.plot_input_sig_freq.setYLink(self.gui.plot_output_sig_freq)
        pass

    
    def apply_optimizations_to_views(self):
        for view in self.views:
            view.getPlotItem().setDownsampling(auto=True, ds = 1, mode = 'subsample')
            view.getPlotItem().setClipToView(True)

    # Function to change playback speed
    def change_speed(self):
        self.speed_state +=1
        
        match self.speed_state:
            case 1:
                self.gui.btn_speed.setText('x1.0')
                self.media_player_input.setPlaybackRate(1.0)
            case 2:
                self.gui.btn_speed.setText('x1.25')
                self.media_player_input.setPlaybackRate(1.25)
            case 3:
                self.gui.btn_speed.setText('x1.5')
                self.media_player_input.setPlaybackRate(1.5)
            case 4:
                self.gui.btn_speed.setText('x1.75')
                self.media_player_input.setPlaybackRate(1.75)
            case 5:
                self.gui.btn_speed.setText('x2.0')
                self.media_player_input.setPlaybackRate(2.0)
            case 6:
                self.gui.btn_speed.setText('x0.5')
                self.media_player_input.setPlaybackRate(0.5)
           
            case 7: # Case 7 to loop back to case 1
                self.gui.btn_speed.setText('x1.0')
                self.media_player_input.setPlaybackRate(1.0)
                self.playback_speed = 1.0
                self.speed_state = 1
            
            case _: # Default Case
                self.speed_state = 1
                self.gui.btn_speed.setText('x1.0')
                self.playback_speed = 1.0
                
                
    def update_player_position(self):
        
        # Set media player positions to value of seeker bar on the plot
        self.media_player_input.setPosition(int(self.medPlayer_seeker.value()))
        self.media_player_output.setPosition(int(self.medPlayer_seeker.value()))
        
        
        
          
    #TODO - CHANGE INTO ONE FUNCTION TO AVOID REPITITION

    # Function to show specified sliders and change their labels
    def modifiy_sliders(self, start_index, end_index, new_slider_name):
        
        # for slider in self.slider_wgts[start_index:end_index]:
        #     slider.setVisible(True)
        for i, widget in enumerate(self.slider_wgts):
            
            # Only show slider widgets that are in given range
            if i in range(start_index, end_index):
                widget.setVisible(True)
            else:
                widget.setVisible(False)
            
            # Set slider label text
            widget.findChild(QtWidgets.QLabel).setText(f"{new_slider_name} {i+1}")
    
    # "Mode Changing" methods
    def change_mode_uniform(self):
        self.modifiy_sliders(0, 10, 'Slider')            
        
    def change_mode_instruments(self):
        self.modifiy_sliders(0, 4, 'Instrument')
        
    def change_mode_animals(self):
        self.modifiy_sliders(0, 4, 'Animal')
    
    def change_mode_ECG(self):
        self.modifiy_sliders(0, 3, 'Arrithmiya')
   
    
    def switch_modes(self):
        mode = self.gui.cmbx_mode_selection.currentText()
        print(mode)
        
        match mode:
            
            case "Uniform Range Mode":
                self.change_mode_uniform()
                self.clear_graphs()
                
            case "Musical Instruments Mode":
                self.change_mode_instruments()
                self.clear_graphs()
                
            case "Animal Sounds Mode":
                self.change_mode_animals()
                self.clear_graphs()
                
            case "ECG Abnormalities Mode":
                self.change_mode_ECG()
                self.clear_graphs()
                
            case _:
                print ("Default Case")
    
    def clear_graphs(self):
        self.reset_sliders()
        for i in range(10):
            self.sliders_freqs[i].setText(str(0))

        self.gui.plot_input_sig_time.clear()
        self.gui.plot_output_sig_time.clear()

        self.gui.plot_input_sig_freq.clear()
        self.gui.plot_output_sig_freq.clear()

        self.gui.plot_input_sig_spect.clear()  
        self.gui.plot_output_sig_spect.clear()

        


    def reset_sliders(self):
        for i in range(10):
            self.sliders[i].setValue(0)
            self.sliders_gains[i].setText(str(0))

    
    

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
        self.gui.wgt_multWindow_amp.setVisible(False)
        self.gui.slider_amplitude_2.setEnabled(False)

        if selected_item == "Rectangle":
            self.mult_window = "rectangle"
            self.plot_multWindow_rectangle()

        elif selected_item == "Hamming":
            self.mult_window = "hamming"
            self.plot_multWindow_hamming()

        elif selected_item == "Hanning":
            self.mult_window = "hanning"
            self.plot_multWindow_hanning()

        elif selected_item == "Gaussian":
            self.mult_window = "gaussian"
            self.gui.wgt_multWindow_amp.setVisible(True)
            self.gui.slider_amplitude_2.setEnabled(True)
            self.plot_multWindow_gaussian()

        print(f"Selected window: {self.mult_window}")

    def plot_multWindow_rectangle(self):
        # Plot rectangle window in self.gui.plot_multWindow
        x = np.linspace(0, 1, 1000)
        y = np.ones_like(x)
        self.gui.plot_multWindow.clear()
        self.gui.plot_multWindow.plot(x, y, pen="r")

    def plot_multWindow_hamming(self):
        # Plot hamming window in self.gui.plot_multWindow
        x = np.linspace(0, 1, 1000)
        y = np.hamming(len(x))
        self.gui.plot_multWindow.clear()
        self.gui.plot_multWindow.plot(x, y, pen="r")

    def plot_multWindow_hanning(self):
        # Plot hanning window in self.gui.plot_multWindow
        x = np.linspace(0, 1, 1000)
        y = np.hanning(len(x))
        self.gui.plot_multWindow.clear()
        self.gui.plot_multWindow.plot(x, y, pen="r")

    def plot_multWindow_gaussian(self):
        # Plot gaussian window in self.gui.plot_multWindow
        x = np.linspace(0, 1, 1000)
        # Assuming self.gui.slider_amplitude_2.value() is the amplitude parameter
        amplitude = self.gui.slider_amplitude_2.value()
        y = amplitude * np.exp(-(x - 0.5)**2 / (2 * 0.1**2))  # Adjust the Gaussian shape as needed
        self.gui.plot_multWindow.clear()
        self.gui.plot_multWindow.plot(x, y, pen="r")



    
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
    
    # Sets the media file to be played by the player
    def load_media_file(self,media: QMediaPlayer, path):
            media_content = QMediaContent(QUrl.fromLocalFile(path))
            media.setMedia(media_content)
        
    # Governs Playing and pausing
    def play_file(self, media: QMediaPlayer):
        if self.sample_rate is not None:
            if media.state() == QMediaPlayer.State.PlayingState:
                media.pause()
            else:            
                media.play()

            

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
        # Take the inverse Fourier transform to get the modified time-domain signal
        modified_signal = np.fft.irfft(self.data_modified_fft)

           # Convert the data to the appropriate integer type for wavfile.write
        modified_signal = modified_signal.astype(np.int16)

        # Print information about the modified signal
        print("Modified Signal Info:")
        print("Shape:", modified_signal.shape)
        print("Min value:", np.min(modified_signal))
        print("Max value:", np.max(modified_signal))
        
        self.gui.slider1.
        fileIO = io.BytesIO()

        # Write the WAV file
        # wavfile.write('output.wav', self.sample_rate, modified_signal)
        wavfile.write(fileIO, self.sample_rate, modified_signal)
        
        buffer = QtCore.QBuffer()
        buffer.setData(fileIO.getvalue())
        buffer.open(QtCore.QIODevice.ReadOnly)
        
        
        # # Load temporary output wav file
        # self.load_media_file(self.media_player_output, 'output.wav')

        print("Output file is saved")




    def open_wav_file(self):
        try:
            files_name = QFileDialog.getOpenFileName(self, 'Open WAV File', os.getenv('HOME'), "WAV files (*.wav)")
            self.path = files_name[0]
            if self.path:
                signal, sample_rate = librosa.load(self.path)
                
                # TODO - Re-add the new loading method
                # Load media file into media player for input
                self.load_media_file(self.media_player_input, self.path)

                # sample_rate, signal = wavfile.read(self.path)
                self.data = signal
                self.sample_rate = sample_rate


                self.data_fft = np.fft.rfft(signal)
                self.frequencies = np.fft.rfftfreq(len(signal), 1 / sample_rate)


                self.data_modified = self.data    
                self.data_modified_fft = self.data_fft
                self.data_modified_frequencies = self.frequencies

                self.section_width = len(self.frequencies) // 10
                for i in range(10):
                    start_idx = i * self.section_width
                    end_idx = (i + 1) * self.section_width
                    self.data_ranges[i] = [start_idx, end_idx]


                self.plot_on_main(self.data, self.data_fft, self.frequencies)
                self.plot_on_secondary(self.data_modified, self.data_modified_fft, self.data_modified_frequencies)

                self.plot_spectrogram_main()
                self.plot_spectrogram_secondary()

                self.set_bands_freq_sliders()

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



    
    # def plot_loaded_signal(self):
    #     self.plot_on_main()
    #     # self.plot_on_secondary()
    #     # self.gui.plot_input_sig_time.addItem(self.medPlayer_seeker)
    #     # self.gui.plot_output_sig_time.addItem(self.medPlayer_seeker)



    def plot_on_main(self, data, data_fft, freq):
            self.gui.plot_input_sig_time.clear()
            self.gui.plot_input_sig_freq.clear()

            self.gui.plot_input_sig_time.plot(data, pen="r")
            self.gui.plot_input_sig_freq.plot(freq, np.abs(data_fft), pen="r")


    def plot_on_secondary(self, data, data_fft, freq):
        self.gui.plot_output_sig_time.clear()
        self.gui.plot_output_sig_freq.clear()

        self.gui.plot_output_sig_time.plot(data, pen="r")
        self.gui.plot_output_sig_freq.plot(freq, np.abs(data_fft), pen="r")
    


    def set_bands_freq_sliders(self):
        for i in range(10):
            self.sliders_freqs[i].setText(str((self.data_ranges[i][1])))

        
    def connect_sliders(self, index):
        self.sliders[index].valueChanged.connect(lambda: self.mult_freqs(index))


    def mult_freqs(self, index):
        self.data_modified_fft = self.multiply_fft(
            self.data_modified_fft,
            self.data_ranges[index][0],
            self.data_ranges[index][1],
            10**(self.sliders[index].value()/20),
            std_gaussian=self.section_width / self.std,
            mult_window=self.mult_window
        )
        self.sliders_gains[index].setText(str(self.sliders[index].value()))

        self.data_modified = np.fft.irfft(self.data_modified_fft)
        
        self.plot_on_secondary(self.data_modified, self.data_modified_fft, self.data_modified_frequencies)
        self.plot_spectrogram_secondary()

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
