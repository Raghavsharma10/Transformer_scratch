def time_bins(header):
  '''
  Returns the time-axis lower bin edge values for the spectrogram.
  '''
  return np.arange(header['number_of_half_frames'], dtype=np.float64)*constants.bins_per_half_frame\
  *(1.0 - header['over_sampling']) / header['subband_spacing_hz']