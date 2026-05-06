def file_checksum(fname):
    """Return md5 checksum of file.

    Note: only works for files < 4GB.

    Parameters
    ----------
    filename : str
        File used to calculate checksum.

    Returns
    -------
        checkum : str
    """
    size = os.path.getsize(fname)
    with open(fname, "r+") as f:
        checksum = hashlib.md5(mmap.mmap(f.fileno(), size)).hexdigest()
    return checksum