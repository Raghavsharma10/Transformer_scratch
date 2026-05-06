def http_download(url, download_file,
                  overwrite=False, raise_http_exception=True):
    """Download a file over HTTP(S).
    
    See: http://stackoverflow.com/a/13137873/5651021 

    Parameters
    ----------
    url : str
        The URL.
    download_file : str
        The path of the local file to write to.
    overwrite : bool, optional
        Whether to overwrite an existing file (if present). [False]
    raise_http_exception : bool, optional
        Whether to raise an exception if there is an HTTP error. [True]

    Raises
    ------
    OSError
        If the file already exists and overwrite is set to False.
    `requests.HTTPError`
        If an HTTP error occurred and `raise_http_exception` was set to `True`.
    """

    assert isinstance(url, (str, _oldstr))
    assert isinstance(download_file, (str, _oldstr))
    assert isinstance(overwrite, bool)
    assert isinstance(raise_http_exception, bool)

    u = urlparse.urlparse(url)
    assert u.scheme in ['http', 'https']

    if os.path.isfile(download_file) and not overwrite:
        raise OSError('File "%s" already exists!' % download_file)
    
    r = requests.get(url, stream=True)
    if raise_http_exception:
        r.raise_for_status()
    if r.status_code == 200:
        with open(download_file, 'wb') as fh:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, fh)
        _logger.info('Downloaded file "%s".', download_file)
    else:
        _logger.error('Failed to download url "%s": HTTP status %d/%s',
                      url, r.status_code, r.reason)