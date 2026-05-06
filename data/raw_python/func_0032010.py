def gzip_decode(data, max_decode=20971520):
    """gzip encoded data -> unencoded data

    Decode data using the gzip content encoding as described in RFC 1952
    """
    if not gzip:
        raise NotImplementedError
    f = StringIO.StringIO(data)
    gzf = gzip.GzipFile(mode="rb", fileobj=f)
    try:
        if max_decode < 0: # no limit
            decoded = gzf.read()
        else:
            decoded = gzf.read(max_decode + 1)
    except IOError:
        raise ValueError("invalid data")
    f.close()
    gzf.close()
    if max_decode >= 0 and len(decoded) > max_decode:
        raise ValueError("max gzipped payload length exceeded")
    return decoded