def loads(ast_str):
    """
    Create a Trace from a suitably compressed string.
    """
    data = zlib.decompress(ast_str)
    li = struct.unpack('<L', data[0:4])[0]
    lt = struct.unpack('<L', data[4:8])[0]
    n = data[8:8 + li].decode('utf-8')
    t = np.fromstring(data[8 + li:8 + li + lt])
    d = np.fromstring(data[8 + li + lt:])

    return Trace(d, t, name=n)