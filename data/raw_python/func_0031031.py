def get_file_checksums(url, ftp=None):
    """Download and parse an Ensembl CHECKSUMS file and obtain checksums.

    Parameters
    ----------
    url : str
        The URL of the CHECKSUM file.
    ftp : `ftplib.FTP` or `None`, optional
        An FTP connection.
    
    Returns
    -------
    `collections.OrderedDict`
        An ordered dictionary containing file names as keys and checksums as
        values.

    Notes
    -----
    The checksums contains in Ensembl CHECKSUM files are obtained with the
    UNIX `sum` command.
    """
    assert isinstance(url, (str, _oldstr))
    if ftp is not None:
        assert isinstance(ftp, ftplib.FTP)

    # open FTP connection if necessary
    close_connection = False
    ftp_server = 'ftp.ensembl.org'
    ftp_user = 'anonymous'
    if ftp is None:
        ftp = ftplib.FTP(ftp_server)
        ftp.login(ftp_user)
        close_connection = True    
    
    # download and parse CHECKSUM file
    data = []
    ftp.retrbinary('RETR %s' % url, data.append)
    data = ''.join(d.decode('utf-8') for d in data).split('\n')[:-1]
    file_checksums = OrderedDict()
    for d in data:
        file_name = d[(d.rindex(' ') + 1):]
        sum_ = int(d[:d.index(' ')])
        file_checksums[file_name] = sum_
    
    logger.debug('Obtained checksums for %d files', len(file_checksums))

    # close FTP connection if we opened it
    if close_connection:
        ftp.close()
    
    return file_checksums