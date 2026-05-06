def periods(dts, phi=0.0):
    """For an ensemble of oscillators, return the set of periods lengths of 
    all successive oscillations of all oscillators.

    An individual oscillation is defined to start and end when the phase 
    passes phi (by default zero) after completing a full cycle.

    If the timeseries of an oscillator phase begins (or ends) exactly at phi, 
    then the first (or last) oscillation will be included.

    Arguments:
      dts (DistTimeseries): where dts.shape[1] is 1 (single output variable
        representing phase) and axis 2 ranges over multiple realizations of
        the oscillator.

      phi=0.0: float
          A single oscillation starts and ends at phase phi (by default zero).
    """
    vperiods = distob.vectorize(analyses1.periods)
    all_periods = vperiods(dts, phi)
    if hasattr(type(all_periods), '__array_interface__'):
        return np.ravel(all_periods)
    else:
        return np.hstack([distob.gather(plist) for plist in all_periods])