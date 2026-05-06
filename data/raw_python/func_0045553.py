def difference(arr, n=1, axis=0, **kwargs):
  '''
  Assuming that `arr` is a 2D spectrogram returned by
  ibmseti.dsp.raw_to_spectrogram(data), this function
  uses the Numpy.diff function to calculate the nth
  difference along either time or frequency.

  If axis = 0 and n=1, then the first difference is taken
  between subsequent time samples

  If axis = 1 and n=1, then the first difference is taken
  between frequency bins.

  For example:

    //each column is a frequency bin
    x = np.array([
       [ 1,  3,  6, 10],   //each row is a time sample
       [ 0,  5,  6,  8],
       [ 2,  6,  9, 12]])

    ibmseti.features.first_difference(x, axis=1)
    >>> array([[2, 3, 4],
               [5, 1, 2],
               [4, 3, 3]])

    ibmseti.features.first_difference(x, axis=0)
    >>> array([[-1,  2,  0, -2],
               [ 2,  1,  3,  4]])

  '''
  return np.diff(arr, n=n, axis=axis, **kwargs)