def solidangle(twotheta, sampletodetectordistance, pixelsize=None):
    """Solid-angle correction for two-dimensional SAS images

    Inputs:
        twotheta: matrix of two-theta values
        sampletodetectordistance: sample-to-detector distance
        pixelsize: the pixel size in mm

    The output matrix is of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by it.
    """
    if pixelsize is None:
        pixelsize = 1
    return sampletodetectordistance ** 2 / np.cos(twotheta) ** 3 / pixelsize ** 2