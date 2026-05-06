def solidangle_errorprop(twotheta, dtwotheta, sampletodetectordistance, dsampletodetectordistance, pixelsize=None):
    """Solid-angle correction for two-dimensional SAS images with error propagation

    Inputs:
        twotheta: matrix of two-theta values
        dtwotheta: matrix of absolute error of two-theta values
        sampletodetectordistance: sample-to-detector distance
        dsampletodetectordistance: absolute error of sample-to-detector distance

    Outputs two matrices of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by the first one. The second one is the propagated
        error of the first one.
    """
    SAC = solidangle(twotheta, sampletodetectordistance, pixelsize)
    if pixelsize is None:
        pixelsize = 1
    return (SAC,
            (sampletodetectordistance * (4 * dsampletodetectordistance ** 2 * np.cos(twotheta) ** 2 +
                                        9 * dtwotheta ** 2 * sampletodetectordistance ** 2 * np.sin(twotheta) ** 2) ** 0.5
             / np.cos(twotheta) ** 4) / pixelsize ** 2)