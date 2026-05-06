def get_mean_threshold_from_calibration(gdac, mean_threshold_calibration):
    '''Calculates the mean threshold from the threshold calibration at the given gdac settings. If the given gdac value was not used during caluibration
    the value is determined by interpolation.

    Parameters
    ----------
    gdacs : array like
        The GDAC settings where the threshold should be determined from the calibration
    mean_threshold_calibration : pytable
        The table created during the calibration scan.

    Returns
    -------
    numpy.array, shape=(len(gdac), )
        The mean threshold values at each value in gdacs.
    '''
    interpolation = interp1d(mean_threshold_calibration['parameter_value'], mean_threshold_calibration['mean_threshold'], kind='slinear', bounds_error=True)
    return interpolation(gdac)