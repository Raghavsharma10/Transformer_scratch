def compamp_to_ac(compamp, window=np.hanning):  # convert single or multi-subband compamps into autocorrelation waterfall

  '''
  Adapted from Gerry Harp at SETI.
  
  '''
  header = compamp.header()
 
  cdata = compamp.complex_data()

  #Apply Windowing and Padding
  cdata = np.multiply(cdata, window(cdata.shape[2]))  # window for smoothing sharp time series start/end in freq. dom.
  cdata_normal = cdata - cdata.mean(axis=2)[:, :, np.newaxis]  # zero mean, does influence a minority of lines in some plots

  cdata = np.zeros((cdata.shape[0], cdata.shape[1], 2 * cdata.shape[2]), complex)
  cdata[:, :, cdata.shape[2]/2:cdata.shape[2] + cdata.shape[2]/2] = cdata_normal  # zero-pad to 2N

  #Perform Autocorrelation
  cdata = np.fft.fftshift(np.fft.fft(cdata), 2)  # FFT all blocks separately and arrange correctly
  cdata = cdata.real**2 + cdata.imag**2  # FFT(AC(x)) = FFT(x)FFT*(x) = abs(x)^2
  cdata = np.fft.ifftshift(np.fft.ifft(cdata), 2)  # AC(x) = iFFT(abs(x)^2) and arrange correctly
  cdata = np.abs(cdata)  # magnitude of AC

  # normalize each row to sqrt of AC triangle
  cdata = np.divide(cdata, np.sqrt(np.sum(cdata, axis=2))[:, :, np.newaxis])  

  return cdata