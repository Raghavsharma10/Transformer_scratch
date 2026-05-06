def convert_from_file(file):
    """
    Reads the content of file in IDX format, converts it into numpy.ndarray and
    returns it.
    file is a file-like object (with read() method) or a file name.
    """
    if isinstance(file, six_string_types):
        with open(file, 'rb') as f:
            return _internal_convert(f)
    else:
        return _internal_convert(file)