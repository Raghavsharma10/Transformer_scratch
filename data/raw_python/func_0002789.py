def _doAtomicFileCreation(filePath):
    """Tries to atomically create the requested file."""
    try:
        _os.close(_os.open(filePath, _os.O_CREAT | _os.O_EXCL))
        return True
    except OSError as e:
        if e.errno == _errno.EEXIST:
            return False
        else:
            raise e