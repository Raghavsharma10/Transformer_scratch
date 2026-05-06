def fourier_to_time(fcdata, norm=None):
  '''
  Converts the data from 2D fourier space signal to a 1D time-series.

  fcdata: Complex fourier spectrum as a 2D array, The axis=0 is for each "half frame", and axis=1 contains the
  fourier-space data for that half frame. Typically there are 129 "half frames" in the data. 
  Furthermore, it's assumed that fftshift has placed the central frequency at the center of axis=1.

  norm: None or "ortho" -- see Numpy FFT Normalization documentation. Default is None.

  Usage:

    aca = ibmseti.compamp.Compamp(raw_data)
    cdata = aca.complex_data()  #cdata is a 3D numpy array in the time domain.
    #can manipulate cdata in time-space if desired (use various windowing functions, for example)
    fcdata = ibmseti.dsp.complex_to_fourier(cdata, aca.header()['over_sampling'])
    fcdata_2d = ibmseti.dsp.reshape_to_2d(fcdata)
    tcdata_1d = ibmseti.dsp.fourier_to_time(fcdata_2d)


  One can recover the Fourier Spectrum of cdata_1d by:

    cdata_2d = cdata_1d.reshape(aca.header()['number_of_half_frames'], int(aca.header()['number_of_subbands'] * ibmseti.constants.bins_per_half_frame*(1 - aca.header()['over_sampling'])))
    fcdata_2d_v2 = np.fft.fftshift(np.fft.fft(cdata_2d), 1)

    #fcdata_2d_v2 and fcdata_2d should be the same
    np.sum(np.sum(fcdata_2d - fcdata_2d_v2))  # should equal to approximately 0

  '''

  return np.fft.ifft(np.fft.ifftshift(fcdata, 1),norm=norm).reshape(fcdata.shape[0] * fcdata.shape[1])