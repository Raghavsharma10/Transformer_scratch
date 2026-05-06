def variability_fp(ts, freqs=None, ncycles=6, plot=True):
    """Example variability function.
    Gives two continuous, time-resolved measures of the variability of a
    time series, ranging between -1 and 1. 
    The two measures are based on variance of the centroid frequency and 
    variance of the height of the spectral peak, respectively.
    (Centroid frequency meaning the power-weighted average frequency)
    These measures are calculated over sliding time windows of variable size.
    See also: Blenkinsop et al. (2012) The dynamic evolution of focal-onset 
              epilepsies - combining theoretical and clinical observations
    Args:
      ts  Timeseries of m variables, shape (n, m). Assumed constant timestep.
      freqs   (optional) List of frequencies to examine. If None, defaults to
              50 frequency bands ranging 1Hz to 60Hz, logarithmically spaced.
      ncycles  Window size, in number of cycles of the centroid frequency.
      plot  bool  Whether to display the output

    Returns:
      variability   Timeseries of shape (n, m, 2)  
                    variability[:, :, 0] gives a measure of variability 
                    between -1 and 1 based on variance of centroid frequency.
                    variability[:, :, 1] gives a measure of variability 
                    between -1 and 1 based on variance of maximum power.
    """
    if ts.ndim <= 2:
        return analyses1.variability_fp(ts, freqs, ncycles, plot)
    else:
        return distob.vectorize(analyses1.variability_fp)(
                ts, freqs, ncycles, plot)