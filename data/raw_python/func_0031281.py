def ftp_download(url, download_file, if_exists='error',
                 user_name='anonymous', password='', blocksize=4194304):
    """Downloads a file from an FTP server.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    download_file : str
        The path of the local file to download to. 
    if_exists : str, optional
        Desired behavior when the download file already exists. One of:
          'error'     - Raise an OSError
          'skip'      - Do nothing, only report a warning.
          'overwrite' - Overwrite the file. reporting a warning.
        Default: 'error'.
    user_name : str, optional
        The user name to use for logging into the FTP server. ['anonymous']
    password : str, optional
        The password to use for logging into the FTP server. ['']
    blocksize : int, optional
        The blocksize (in bytes) to use for downloading. [4194304]

    Returns
    -------
    None
    """
    assert isinstance(url, (str, _oldstr))
    assert isinstance(download_file, (str, _oldstr))
    assert isinstance(if_exists, (str, _oldstr))
    assert isinstance(user_name, (str, _oldstr))
    assert isinstance(password, (str, _oldstr))

    u = urlparse.urlparse(url)
    assert u.scheme == 'ftp'

    if if_exists not in ['error', 'skip', 'overwrite']:
        raise ValueError('"if_exists" must be "error", "skip", or "overwrite" '
                         '(was: "%s").', str(if_exists))

    if os.path.isfile(download_file):
        if if_exists == 'error':
            raise OSError('File "%s" already exists.' % download_file)
        elif if_exists == 'skip':
            _logger.warning('File "%s" already exists. Skipping...',
                            download_file)
            return
        else:
            _logger.warning('Overwriting file "%s"...', download_file)

    ftp_server = u.netloc
    ftp_path = u.path

    if six.PY3:
        with ftplib.FTP(ftp_server) as ftp:
            ftp.login(user_name, password)
            with open(download_file, 'wb') as ofh:
                ftp.retrbinary('RETR %s' % ftp_path,
                               callback=ofh.write, blocksize=blocksize)
    else:
        ftp = ftplib.FTP(ftp_server)
        ftp.login(user_name, password)
        with open(download_file, 'wb') as ofh:
            ftp.retrbinary('RETR %s' % ftp_path,
                           callback=ofh.write, blocksize=blocksize)
        ftp.close()

    _logger.info('Downloaded file "%s" over FTP.', download_file)