def get_hit_rate_correction(gdacs, calibration_gdacs, cluster_size_histogram):
    '''Calculates a correction factor for single hit clusters at the given GDACs from the cluster_size_histogram via cubic interpolation.

    Parameters
    ----------
    gdacs : array like
        The GDAC settings where the threshold should be determined from the calibration
    calibration_gdacs : array like
        GDAC settings used during the source scan for the cluster size calibration.
    cluster_size_histogram : numpy.array, shape=(80,336,# of GDACs during calibration)
        The calibration array

    Returns
    -------
    numpy.array, shape=(80,336,# of GDACs during calibration)
        The threshold values for each pixel at gdacs.
    '''
    logging.info('Calculate the correction factor for the single hit cluster rate at %d given GDAC settings', len(gdacs))
    if len(calibration_gdacs) != cluster_size_histogram.shape[0]:
        raise ValueError('Length of the provided pixel GDACs does not match the dimension of the cluster size array')
    hist_sum = np.sum(cluster_size_histogram, axis=1)
    hist_rel = cluster_size_histogram / hist_sum[:, np.newaxis].astype('f4') * 100.
    maximum_rate = np.amax(hist_rel[:, 1])
    correction_factor = maximum_rate / hist_rel[:, 1]
    # sort arrays since interpolate does not work otherwise
    calibration_gdacs_sorted = np.array(calibration_gdacs)
    correction_factor_sorted = correction_factor[np.argsort(calibration_gdacs_sorted)]
    calibration_gdacs_sorted = np.sort(calibration_gdacs_sorted)
    interpolation = interp1d(calibration_gdacs_sorted.tolist(), correction_factor_sorted.tolist(), kind='cubic', bounds_error=True)
    return interpolation(gdacs)