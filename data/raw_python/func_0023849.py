def convert_to_string(ndarr):
    """
    Writes the contents of the numpy.ndarray ndarr to bytes in IDX format and
    returns it.
    """
    with contextlib.closing(BytesIO()) as bytesio:
        _internal_write(bytesio, ndarr)
        return bytesio.getvalue()