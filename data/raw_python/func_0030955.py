def read_single(path, encoding = 'UTF-8'):
    """ Reads the first column of a tab-delimited text file.

    The file can either be uncompressed or gzip'ed.

    Parameters
    ----------
    path: str
        The path of the file.
    enc: str
        The file encoding.

    Returns
    -------
    List of str
        A list containing the elements in the first column.

    """
    assert isinstance(path, (str, _oldstr))
    data = []
    with smart_open_read(path, mode='rb', try_gzip=True) as fh:
        reader = csv.reader(fh, dialect='excel-tab', encoding=encoding)
        for l in reader:
            data.append(l[0])
    return data