def azimintpix(data, dataerr, bcx, bcy, mask=None, Ntheta=100, pixmin=0,
               pixmax=np.inf, returnmask=False, errorpropagation=2):
    """Azimuthal integration (averaging) on the detector plane

    Inputs:
        data: scattering pattern matrix (np.ndarray, dtype: np.double)
        dataerr: error matrix (np.ndarray, dtype: np.double; or None)
        bcx, bcy: beam position, counting from 1
        mask: mask matrix (np.ndarray, dtype: np.uint8)
        Ntheta: Number of points in the abscissa (azimuth angle)
        pixmin: smallest distance from the origin in pixels
        pixmax: largest distance from the origin in pixels
        returnmask: if the effective mask matrix is to be returned

    Outputs: theta, Intensity, [Error], Area, [mask]
        Error is only returned if dataerr is not None
        mask is only returned if returnmask is True

    Relies heavily (completely) on azimint().
    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.double)
    if isinstance(dataerr, np.ndarray):
        dataerr = dataerr.astype(np.double)
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    return azimint(data, dataerr, -1, -1,
                   - 1, bcx, bcy, mask, Ntheta, pixmin,
                   pixmax, returnmask, errorpropagation)