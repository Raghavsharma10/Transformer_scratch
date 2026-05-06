def readmask(filename, fieldname=None):
    """Try to load a maskfile from a matlab(R) matrix file

    Inputs
    ------
    filename: string
        the input file name
    fieldname: string, optional
        field in the mat file. None to autodetect.

    Outputs
    -------
    the mask in a numpy array of type np.uint8
    """
    f = scipy.io.loadmat(filename)
    if fieldname is not None:
        return f[fieldname].astype(np.uint8)
    else:
        validkeys = [
            k for k in list(f.keys()) if not (k.startswith('_') and k.endswith('_'))]
        if len(validkeys) < 1:
            raise ValueError('mask file contains no masks!')
        if len(validkeys) > 1:
            raise ValueError('mask file contains multiple masks!')
        return f[validkeys[0]].astype(np.uint8)