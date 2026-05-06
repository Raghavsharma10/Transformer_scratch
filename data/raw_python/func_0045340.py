def _spec_fft(self, complex_data):
    '''
    Calculates the DFT of the complex_data along axis = 1.  This assumes complex_data is a 2D array.

    This uses numpy and the code is straight forward
    np.fft.fftshift( np.fft.fft(complex_data), 1)

    Note that we automatically shift the FFT frequency bins so that along the frequency axis, 
    "negative" frequencies are first, then the central frequency, followed by "positive" frequencies.
    '''
    return np.fft.fftshift( np.fft.fft(complex_data), 1)