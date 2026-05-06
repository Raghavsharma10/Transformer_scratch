def total_variation(arr):
  '''
  If arr is a 2D array (N X M), assumes that arr is a spectrogram with time along axis=0.

  Calculates the 1D total variation in time for each frequency and returns an array
  of size M.

  If arr is a 1D array, calculates total variation and returns a scalar.

  Sum ( Abs(arr_i+1,j  - arr_ij) )

  If arr is a 2D array, it's common to take the mean of the resulting M-sized array
  to calculate a scalar feature.
  '''
  return np.sum(np.abs(np.diff(arr, axis=0)), axis=0)