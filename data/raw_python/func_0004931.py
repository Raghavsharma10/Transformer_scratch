def writeint2dnorm(filename, Intensity, Error=None):
    """Save the intensity and error matrices to a file

    Inputs
    ------
    filename: string
        the name of the file
    Intensity: np.ndarray
        the intensity matrix
    Error: np.ndarray, optional
        the error matrix (can be ``None``, if no error matrix is to be saved)

    Output
    ------
    None
    """
    whattosave = {'Intensity': Intensity}
    if Error is not None:
        whattosave['Error'] = Error
    if filename.upper().endswith('.NPZ'):
        np.savez(filename, **whattosave)
    elif filename.upper().endswith('.MAT'):
        scipy.io.savemat(filename, whattosave)
    else:  # text file
        np.savetxt(filename, Intensity)
        if Error is not None:
            name, ext = os.path.splitext(filename)
            np.savetxt(name + '_error' + ext, Error)