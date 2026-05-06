def maximum_variation(arr):
  '''
  return np.max(arr, axis=0) - np.min(arr, axis=0)

  If `arr` is a 1D array, a scalar is returned.

  If `arr` is a 2D array (N x M), an array of length M is returned.
  '''
  return np.max(arr, axis=0) - np.min(arr, axis=0)