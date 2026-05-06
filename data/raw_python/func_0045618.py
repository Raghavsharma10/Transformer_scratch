def complex_to_fourier(cdata, over_sampling, norm=None):
  '''
  cdata: 3D complex data (shaped by subbands and half_frames, as returned from Compamp.complex_data())
  over_sampling: The fraction of oversampling across subbands (typically 0.25)
  norm: None or "ortho" -- see Numpy FFT Normalization documentation. Default is None.

  returns the signal in complex fourier space. The output fourier data are shifted so the central frequency
  is at the center of the values. All over-sampled frequencies have been removed so that all frequency bins
  can be properly arranged next to each other. 
  '''
  
  # FFT all blocks separately and rearrange output
  fftcdata = np.fft.fftshift(np.fft.fft(cdata, norm=norm), 2)  
  
  # slice out oversampled frequencies
  if over_sampling > 0:
    fftcdata = fftcdata[:, :, int(cdata.shape[2]*over_sampling/2):-int(cdata.shape[2]*over_sampling/2)] 
    
  return fftcdata