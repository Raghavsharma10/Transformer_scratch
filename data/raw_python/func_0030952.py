def get_file_checksum(path):
    """Get the checksum of a file (using ``sum``, Unix-only).

    This function is only available on certain platforms.

    Parameters
    ----------
    path: str
        The path of the file.

    Returns
    -------
    int
        The checksum.

    Raises
    ------
    IOError
        If the file does not exist.
    """

    if not (sys.platform.startswith('linux') or \
                        sys.platform in ['darwin', 'cygwin']):
        raise OSError('This function is not available on your platform.')

    assert isinstance(path, (str, _oldstr))

    if not os.path.isfile(path): # not a file
        raise IOError('File "%s" does not exist.' %(path))

    # calculate checksum
    sub = subproc.Popen('sum "%s"' %(path), bufsize=-1, shell=True,
                        stdout=subproc.PIPE)
    stdoutdata = sub.communicate()[0]
    assert sub.returncode == 0

    # in Python 3, communicate() returns bytes that need to be decoded
    encoding = locale.getpreferredencoding()
    stdoutstr = str(stdoutdata, encoding=encoding)

    file_checksum = int(stdoutstr.split(' ')[0])
    logger.debug('Checksum of file "%s": %d', path, file_checksum)
    return file_checksum