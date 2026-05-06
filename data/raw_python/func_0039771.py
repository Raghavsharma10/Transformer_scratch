def find_amp_phase(angle, data, npeaks=3, min_amp=None, min_phase=None):
    """Estimate amplitude and phase of an approximately sinusoidal quantity
    using `scipy.optimize.curve_fit`.

    Phase is defined as the angle at which the cosine curve fit reaches its
    first peak. It is assumed that phase is positive. For example:

        data_fit = amp*np.cos(npeaks*(angle - phase)) + mean_data

    Parameters
    ----------
    angle : numpy array
        Time series of angle values in radians
    data : numpy array
        Time series of data to be fit
    npeaks : int
        Number of peaks per revolution, or normalized frequency
    min_phase : float
        Minimum phase to allow for guess to least squares fit

    Returns
    -------
    amp : float
        Amplitude of regressed cosine
    phase : float
        Angle of the first peak in radians
    """
    # First subtract the mean of the data
    data = data - data.mean()
    # Make some guesses for parameters from a subset of data starting at an
    # integer multiple of periods
    if angle[0] != 0.0:
        angle1 = angle[0] + (2*np.pi/npeaks - (2*np.pi/npeaks) % angle[0])
    else:
        angle1 = angle[0]
    angle1 += min_phase
    angle2 = angle1 + 2*np.pi/npeaks
    ind = np.logical_and(angle >= angle1, angle <= angle2)
    angle_sub = angle[ind]
    data_sub = data[ind]
    amp_guess = (data_sub.max() - data_sub.min())/2
    phase_guess = angle[np.where(data_sub == data_sub.max())[0][0]] \
            % (np.pi*2/npeaks)
    # Define the function we will try to fit to
    def func(angle, amp, phase, mean):
        return amp*np.cos(npeaks*(angle - phase)) + mean
    # Calculate fit
    p0 = amp_guess, phase_guess, 0.0
    popt, pcov = curve_fit(func, angle, data, p0=p0)
    amp, phase, mean = popt
    return amp, phase