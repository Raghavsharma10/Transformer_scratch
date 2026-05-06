def find_outliers(data, predictor, sigma,
                  method=mad):
    """find_outliers(data, predictor, sigma, method=mad)

    Returns a boolean array indicating the outliers in the given *data* array.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array containing columns *phase*, *magnitude*, and
        (optional) *error*.
    predictor : object that has "fit" and "predict" methods, optional
        Object which fits the light curve obtained from *data* after rephasing.
    sigma : number
        Outlier cutoff criteria.
    method : function, optional
        Function to score residuals for outlier detection
        (default :func:`plotypus.utils.mad`).

    **Returns**

    out : array-like, shape = data.shape
        Boolean array indicating the outliers in the given *data* array.
    """
    phase, mag, *err = data.T
    residuals = numpy.absolute(predictor.predict(colvec(phase)) - mag)
    outliers = numpy.logical_and((residuals > err[0]) if err else True,
                                 residuals > sigma * method(residuals))

    return numpy.tile(numpy.vstack(outliers), data.shape[1])