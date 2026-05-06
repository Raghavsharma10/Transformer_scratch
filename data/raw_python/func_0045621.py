def reshape_to_2d(arr):
  '''
  Assumes a 3D Numpy array, and reshapes like
  
  arr.reshape((arr.shape[0], arr.shape[1]*arr.shape[2]))

  This is useful for converting processed data from `complex_to_power`
  and from `autocorrelation` into a 2D array for image analysis and display.

  '''
  return arr.reshape((arr.shape[0], arr.shape[1]*arr.shape[2]))