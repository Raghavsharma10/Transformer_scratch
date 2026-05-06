def periods(ts, phi=0.0):
    """For a single variable timeseries representing the phase of an oscillator,
    measure the period of each successive oscillation.

    An individual oscillation is defined to start and end when the phase 
    passes phi (by default zero) after completing a full cycle.

    If the timeseries begins (or ends) exactly at phi, then the first
    (or last) oscillation will be included.

    Arguments: 
      ts: Timeseries (single variable)
          The timeseries of an angle variable (radians)

      phi (float): A single oscillation starts and ends at phase phi (by 
        default zero).
    """
    ts = np.squeeze(ts)
    if ts.ndim <= 1:
        return np.diff(phase_crossings(ts, phi))
    else:
        return np.hstack([ts[...,i].periods(phi) for i in range(ts.shape[-1])])