def read_files(*sources):
    """Construct a generator that yields :class:`~nmrstarlib.nmrstarlib.StarFile` instances.

    :param sources: One or more strings representing path to file(s).
    :return: :class:`~nmrstarlib.nmrstarlib.StarFile` instance(s).
    :rtype: :class:`~nmrstarlib.nmrstarlib.StarFile`
    """
    filenames = _generate_filenames(sources)
    filehandles = _generate_handles(filenames)
    for fh, source in filehandles:
        starfile = nmrstarlib.StarFile.read(fh, source)
        yield starfile