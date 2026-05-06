def complex_to_power(cdata, over_sampling):  
  '''
  cdata: 3D complex data (shaped by subbands and half_frames, as returned from Compamp.complex_data())
  over_sampling: The fraction of oversampling across subbands (typically 0.25). 
  
  returns a 3D spectrogram
  
  Example:
      aca = ibmseti.compamp.Compamp(raw_data)
      cdata = aca.complex_data()
      #can perform any transformations on cdata here, such as applying hanning windows for smoother FFT results.
      #cdata = np.multiply(cdata, np.hanning(constants.bins_per_half_frame))
      power = ibmseti.dsp.complex_to_power(, aca.header()['over_sampling'])
  
  Typically, this 3D spectrogram is rehaped so that the subbands are aligned next to each other
  in a 2D spectrogram
  
      spectrogram = ibmseti.dsp.reshape_to_2d(power)
  '''
  
  fftcdata = complex_to_fourier(cdata, over_sampling)

  # calculate power, normalize and amplify by factor 15 (what is the factor of 15 for?)
  fftcdata = np.multiply(fftcdata.real**2 + fftcdata.imag**2, 15.0/cdata.shape[2])

  return fftcdata