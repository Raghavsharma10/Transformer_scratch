def frequency_bins(header):
  '''
  Returnes the frequency-axis lower bin edge values for the spectrogram. 
  '''

  center_frequency = 1.0e6*header['rf_center_frequency']
  if header["number_of_subbands"] > 1:
    center_frequency += header["subband_spacing_hz"]*(header["number_of_subbands"]/2.0 - 0.5)

  return np.fft.fftshift(\
    np.fft.fftfreq( int(header["number_of_subbands"] * constants.bins_per_half_frame*(1.0 - header['over_sampling'])), \
      1.0/(header["number_of_subbands"]*header["subband_spacing_hz"])) + center_frequency
    )