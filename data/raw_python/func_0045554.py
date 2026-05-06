def projection(arr, axis=0, **kwargs):
  '''
  Assuming that `arr` is a 2D spectrogram returned by
  ibmseti.dsp.raw_to_spectrogram(data), where each row
  of the `arr` is a power spectrum at a particular time,
  this function uses the numpy.sum function to project the
  data onto the time or frequency axis into a 1D array.

  If axis = 0, then the projection is onto the frequency axis
  (the sum is along the time axis)

  If axis = 1, then the projection is onto the time axis.
  (the sum is along the frequency axis)

  For example:

    //each column is a frequency bin
    x = np.array([
       [ 1,  3,  6, 10],   //each row is a time sample
       [ 0,  5,  6,  8],
       [ 2,  6,  9, 12]])

    ibmseti.features.projection(x, axis=1)
    >>> array([20, 19, 29])

    ibmseti.features.projection(x, axis=0)
    >>> array([ 3, 14, 21, 30])

  One interesting kwarg that you may wish to use is `keepdims`.
  See the documentation on numpy.sum for more information.

  '''
  return np.sum(arr, axis=axis, **kwargs)