def get_spectrogram(self):
    '''
    Transforms the input simulated data and computes a standard-sized spectrogram.

    If self.sigProc function is not None, the 2D complex-valued time-series data will 
    be processed with that function before the FFT and spectrogram are calculated. 
    '''

    return self._spec_power(self._spec_fft(  self._sigProc( self._reshape( self.complex_data() )) ))