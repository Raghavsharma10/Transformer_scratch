def _generate_handles(filenames):
    """Open a sequence of filenames one at time producing file objects.
    The file is closed immediately when proceeding to the next iteration.

    :param generator filenames: Generator object that yields the path to each file, one at a time.
    :return: Filehandle to be processed into a :class:`~nmrstarlib.nmrstarlib.StarFile` instance.
    """
    for fname in filenames:
        if nmrstarlib.VERBOSE:
            print("Processing file: {}".format(os.path.abspath(fname)))
        path = GenericFilePath(fname)
        for filehandle, source in path.open():
            yield filehandle, source
            filehandle.close()