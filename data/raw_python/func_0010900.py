def ungzip_data(input_data):
    """Return a string of data after gzip decoding

    :param the input gziped data
    :return  the gzip decoded data"""
    buf = StringIO(input_data)
    f = gzip.GzipFile(fileobj=buf)
    return f