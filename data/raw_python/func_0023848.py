def convert_to_file(file, ndarr):
    """
    Writes the contents of the numpy.ndarray ndarr to file in IDX format.
    file is a file-like object (with write() method) or a file name.
    """
    if isinstance(file, six_string_types):
        with open(file, 'wb') as fp:
            _internal_write(fp, ndarr)
    else:
        _internal_write(file, ndarr)