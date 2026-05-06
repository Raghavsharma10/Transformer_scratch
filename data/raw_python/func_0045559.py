def entropy(p, w):
  '''
  Computes the entropy for a discrete probability distribution function, as
  represented by a histogram, `p`, with bin sizes `w`,

   h_p = Sum -1 * p_i * ln(p_i / w_i)

  Also computes the maximum allowed entropy for a histogram with bin sizes `w`.

    h_max = ln( Sum w_i )

  and returns both as a tuple (h_p , h_max ). The entropy is in 'natural' units.

  Both `p` and `w` must be Numpy arrays.

  If `p` is normalized to 1 ( Sum p_i * w_i = 1), then
  the normalized entropy is equal toh_p / h_max and will
  be in the range [0, 1].

  For example, if `p` is a completely flat PDF (a uniform distribution), then
  the normalized entropy will equal 1, indicating maximum amount of disorder.
  (This is easily shown for the case where w_i = 1.)

  If the `p_i` is zero for all i except j and p_j = 1, then the entropy will be 0,
  indicating no disorder.

  One can use this entropy measurement to search for signals in the spectrogram.
  First we need to build a histogram of the measured power values in the spectrogram.
  This histogram represents an estimate of the probability distribution function of the
  observed power in the spectrogram.

  If the spectrogram is entirely noise, the resulting histogram should be quite flat and
  the normalized entropy ( h_p / h_max ) will approach 1. If there is a significant signal
  in the spectrogram, then the histogram will not be flat and the normalized entropy will
  be less than 1.

  The decision that needs to be made is the number of bins and the bin size. And unfortunately,
  the resulting entropy calculated will depend on the binning.

  Based on testing and interpretibility, we recommend to use a fixed number of bins that either
  span the full range of the power values in the spectrogram (0 to spectrogram.max()),
  or span a fixed range (for example, from 0 to 500).

  For example, you may set the range equal to the range of the values in the spectrogram.

    bin_edges = range(0,int(spectrogram.max()) + 2) #add 1 to round up, and one to set the right bin edge.
    p, _ = np.histogram(spectrogram.flatten(), bins=bin_edges, density=True)
    w = np.diff(bin_edges)
    h_p, h_max = ibmseti.features.entropy(p,w)

  If you choose to fix the range of the histogram, it is highly recommended that you use
  `numpy.clip` to ensure that any of the values in the spectrogram that are greater than
  your largest bin are not thrown away!

  For example, if you decide on a fixed range between 0 and 500, and your spectrogram
  contains a value of 777, the following code would produce a histogram where that 777 value
  is not present in the count.

    bin_edges = range(0,501)
    p, _ = np.histogram(spectrogram.flatten(), bins=bin_edges, density=True)
    w = np.diff(bin_edges)
    h_p, h_max = ibmseti.features.entropy(p,w)

  But if you clip the spectrogram, you can interpret the last bin as being "the number
  of spectrogram values equal to or greater than the lower bin edge".

    bin_edges = range(0,501)
    p, _ = np.histogram(np.clip(spectrogram.flatten(), 0, 500), bins=bin_edges, density=True)
    w = np.diff(bin_edges)
    h_p, h_max = ibmseti.features.entropy(p,w)

  You can also choose to fix the number of bins

    bins = 50
    p, bin_edges = np.histogram(spectrogram.flatten(), bins=bins, density=True)
    w = np.diff(bin_edges)
    h_p, h_max = ibmseti.features.entropy(p,w)

  It is suggested to use any of the following measures as features:

    bin range, spectrogram.min, spectrogram.max, number_of_bins, log(number_of_bins)
    entropy, max_entropy, normalized_entropy.

  Automatic Binning:

  While Numpy and AstroML offer ways of automatically binning the data, it is unclear if this
  is a good approach for entropy calculation -- especially when wishing to compare the value
  across different spectrogram. The automatic binning tends to remove disorder in
  the set of values, making the histogram smoother and more ordered than the data actually are.
  This is true of automatic binning with fixed sizes (such as with the 'rice', and 'fd' options in
  numpy.histogram), or with the variable sized arrays as can be calculated with Bayesian Blocks
  with astroML. However, nothing is ruled out. In preliminary testing,
  the calculated entropy from a histogram calculated with Bayesian Block binning seemed to be more
  sensitive to a simulated signal than using fixed binning. However, it's unclear how to
  interpret the results because "h_p/h_max" *increased* with the presence of a signal and exceeded 1.

  **It is likely that the calculation of h_max is done incorrectly. Please check my work!**

  It may even be that the total number of bins created by the Bayesian Block method would
  be a suitable feature. For a completely flat distribution, there will only be one bin. If the
  data contains significant variation in power levels, the Bayesian Block method will produce more
  bins.  More testing is required and your mileage may vary.

    import astroML.plotting

    bin_edges = astroML.density_estimation.bayesian_blocks(spectrogram.flatten())
    p, _ = np.histogram(spectrogram.flatten(), bins=bin_edges, density=True)
    w = np.diff(bin_edges)

    h_p, h_max = ibmseti.features.entropy(p,w)

  Also to note: Using astroML.density_estimation.bayesian_blocks takes prohibitively long!

  "Entropy" of raw data.

  If `p` is NOT a PDF, then you're on your own to interpret the results. In this case, you
  may set `w` = None and the calculation will assume w_i = 1 for all i.

  For example,

    h_p, _ = ibmseti.features.entropy(spectrogram.flatten(), None)

  '''
  if w is None:
    w = np.ones(len(p))

  h_p = np.sum([-x[0]*math.log(x[0]/x[1]) if x[0] else 0 for x in zip(p, w)])
  h_max = math.log(np.sum(w))

  return h_p, h_max