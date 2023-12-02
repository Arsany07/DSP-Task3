# DSP-Task3

# Signal Equalizer



## Overview

This signal equalizer project provides a user-friendly interface for manipulating and analyzing audio signals. It allows users to open audio files, apply various equalization adjustments, and visualize the effects of these adjustments in real time. The project features:

-   **Audio File Loading:** Users can open audio files in various formats to apply equalization techniques.
    
-   **Equalization Controls:** Users can adjust multiple equalizer sliders to modify the frequency response of the audio signal, enabling them to enhance or suppress specific frequency bands.
    
-   **Synchronized Signal Viewers:** Two signal viewers display the input and output signals simultaneously, allowing users to observe the original signal and the effects of equalization in real time. These viewers are synchronized to ensure that they always show the same time-part of the signal, regardless of scrolling or zooming.
    
-   **Spectrograms:** Two spectrograms, one for the input and one for the output signals, provide a visual representation of the frequency content of the signals. The output spectrogram dynamically updates to reflect the changes made by the equalizer sliders.
    
-   **Spectrogram Toggle:** Users can toggle the visibility of the spectrograms to focus on either the signal viewers or the frequency representations.
    
-   **Equalization Modes:** Four different equalization modes are available:
    
    1.  Uniform Range Mode: Provides a general-purpose equalizer with sliders for adjusting frequencies across the entire audio spectrum.
        
    2.  Musical Instruments Mode: Offers sliders tailored to specific frequency ranges corresponding to different musical instruments, allowing users to enhance or suppress the presence of those instruments in the audio signal.
        
    3.  Animal Sounds Mode: Provides sliders tuned to frequency ranges associated with various animal sounds, enabling users to manipulate the prominence of these sounds in the audio signal.
        
    4.  ECG Abnormalities Mode: Offers sliders designed to detect and highlight potential abnormalities in electrocardiogram (ECG) signals, aiding in the analysis of heart conditions.
        
    
-   **Smoothing Window Customization:** Users can select the type of smoothing window to apply to the equalizer bands, influencing the smoothness of the frequency response adjustments. They can also visually customize the parameters of the smoothing window and observe the effects in real time.
    
-   **Equalizer Application:** Once satisfied with the equalization settings, users can apply the customized equalizer to the audio signal and save the modified audio file.
    

## Usage Instructions

1.  Launch the signal equalizer application.

6.  Select the desired equalization mode from the Mode menu.
    
2.  Open an audio file using the File menu .
    
3.  Adjust the equalizer sliders to modify the frequency response of the audio signal.
    
4.  Observe the effects of equalization in the signal viewers and spectrograms.
    
5.  Toggle the visibility of the spectrograms using the Spectrogram Toggle checkbox.
    
7.  Customize the smoothing window type and parameters using the Smoothing Window panel.
    
9.  Save the modified audio file using the File > Save menu.
    


## Object Names

### Plot names

#### Input Signal Plots

* plot_input_sig_time
* plot_input_sig_freq
* plot_input_sig_spect

#### Output Signal Plots

* plot_output_sig_time
* plot_output_sig_freq
* plot_output_sig_spect
  
###### Multiplication window plot: *plot_multWindow*

#### Playback Buttons

* btn_play
* btn_rewind
* btn_zoom_in
* btn_zoom_out
* btn_pan_left
* btn_pan_right
===

#### Widget names (For hiding and stuff)

* *wgt_sliders*: parent of *wgt_sld_1* till *wgt_sld_10*
* *wgt_multWindow_std* & *wgt_multWindow_amp*

#### Labels that need editing through code

* label_speed: should display "x {slider value divided by 10}" up to one float position
* Slider labels are named label_slider1 to label_slider10
* lbl_value_amp: for the Amplitude slider in multiplication window
* lbl_value_std: for the STD Slider in multiplication window

#### Things that should be set at app runtime

* Text of speed slider (label_speed)
* Hiding the standard deviation widget so it only show when gaussian multiplication window is selected
  (Use: ```wgt_multWindow_std.setVisible(False)``` )

###### All Group boxes are named *grpbx_*
###### All Combo boxes are named *cmbx_*
###### Sliders are named *slider1* till *slider10*, each slider is inside its own widget (idk why I did that)


# TODO for UI

###### Add checkboxes to hide/unhide spectrographs

###### Consider making just one big GraphicsLayoutWidget and Adding multiple plots to it instead of having three plot widgets for input and output
