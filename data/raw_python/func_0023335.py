def load_data_file(fname, directory=None, force_download=False):
    """Get a standard vispy demo data file

    Parameters
    ----------
    fname : str
        The filename on the remote ``demo-data`` repository to download,
        e.g. ``'molecular_viewer/micelle.npy'``. These correspond to paths
        on ``https://github.com/vispy/demo-data/``.
    directory : str | None
        Directory to use to save the file. By default, the vispy
        configuration directory is used.
    force_download : bool | str
        If True, the file will be downloaded even if a local copy exists
        (and this copy will be overwritten). Can also be a YYYY-MM-DD date
        to ensure a file is up-to-date (modified date of a file on disk,
        if present, is checked).

    Returns
    -------
    fname : str
        The path to the file on the local system.
    """
    _url_root = 'http://github.com/vispy/demo-data/raw/master/'
    url = _url_root + fname
    if directory is None:
        directory = config['data_path']
        if directory is None:
            raise ValueError('config["data_path"] is not defined, '
                             'so directory must be supplied')

    fname = op.join(directory, op.normcase(fname))  # convert to native
    if op.isfile(fname):
        if not force_download:  # we're done
            return fname
        if isinstance(force_download, string_types):
            ntime = time.strptime(force_download, '%Y-%m-%d')
            ftime = time.gmtime(op.getctime(fname))
            if ftime >= ntime:
                return fname
            else:
                print('File older than %s, updating...' % force_download)
    if not op.isdir(op.dirname(fname)):
        os.makedirs(op.abspath(op.dirname(fname)))
    # let's go get the file
    _fetch_file(url, fname)
    return fname