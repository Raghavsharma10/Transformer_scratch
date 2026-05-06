def readint2dnorm(filename):
    """Read corrected intensity and error matrices (Matlab mat or numpy npz
    format for Beamline B1 (HASYLAB/DORISIII))

    Input
    -----
    filename: string
        the name of the file

    Outputs
    -------
    two ``np.ndarray``-s, the Intensity and the Error matrices

    File formats supported:
    -----------------------

    ``.mat``
        Matlab MAT file, with (at least) two fields: Intensity and Error

    ``.npz``
        Numpy zip file, with (at least) two fields: Intensity and Error

    other
        the file is opened with ``np.loadtxt``. The error matrix is tried
        to be loaded from the file ``<name>_error<ext>`` where the intensity was
        loaded from file ``<name><ext>``. I.e. if ``somedir/matrix.dat`` is given,
        the existence of ``somedir/matrix_error.dat`` is checked. If not found,
        None is returned for the error matrix.

    Notes
    -----
    The non-existence of the Intensity matrix results in an exception. If the
    Error matrix does not exist, None is returned for it.
    """
    # the core of read2dintfile
    if filename.upper().endswith('.MAT'):  # Matlab
        m = scipy.io.loadmat(filename)
    elif filename.upper().endswith('.NPZ'):  # Numpy
        m = np.load(filename)
    else:  # loadtxt
        m = {'Intensity': np.loadtxt(filename)}
        name, ext = os.path.splitext(filename)
        errorfilename = name + '_error' + ext
        if os.path.exists(errorfilename):
            m['Error'] = np.loadtxt(errorfilename)
    Intensity = m['Intensity']
    try:
        Error = m['Error']
        return Intensity, Error
    except:
        return Intensity, None