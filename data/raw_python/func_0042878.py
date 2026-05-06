def zip2bytes(compressed):
    """
    UNZIP DATA
    """
    if hasattr(compressed, "read"):
        return gzip.GzipFile(fileobj=compressed, mode='r')

    buff = BytesIO(compressed)
    archive = gzip.GzipFile(fileobj=buff, mode='r')
    from pyLibrary.env.big_data import safe_size
    return safe_size(archive)