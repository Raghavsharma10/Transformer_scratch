def moment(arr, moment=1, axis=0, **kwargs):
  '''
  Uses the scipy.stats.moment to calculate the Nth central
  moment about the mean.

  If `arr` is a 2D spectrogram returned by
  ibmseti.dsp.raw_to_spectrogram(data), where each row
  of the `arr` is a power spectrum at a particular time,
  this function, then the Nth moment along each axis
  will be computed.

  If axis = 0, then Nth moment for the data in each
  frequency bin will be computed. (The calculation is done
    *along* the 0th axis, which is the time axis.)

  If axis = 1, then Nth moment for the data in each
  time bin will be computed. (The calculation is done
    *along* the 1st axis, which is the frequency axis.)

  For example, consider the 2nd moment:

    //each column is a frequency bin
    x = array([[  1.,   3.,   6.,  10.], //each row is a time sample
               [  0.,   5.,   6.,   8.],
               [  2.,   6.,   9.,  12.]])

    ibmseti.features.mement(x, moment=2, axis=0) //the returned array is of size 4, the number of columns / frequency bins.
    >>>  array([ 0.66666667,  1.55555556,  2.,  2.66666667])

    ibmseti.features.mement(x, moment=2, axis=1) //the returned array is of size 3, the number of rows / time bins.
    >>>  array([ 11.5 ,  8.6875, 13.6875])

  If `arr` is a 1D array, such as what you'd get if you projected
  the spectrogram onto the time or frequency axis, then you must
  use axis=0.

  '''
  return scipy.stats.moment(arr, moment=moment, axis=axis, **kwargs)