def dumps(asts):
    """
    Create a compressed string from an Trace.
    """
    d = asts.values.tostring()
    t = asts.index.values.astype(float).tostring()
    lt = struct.pack('<L', len(t))
    i = asts.name.encode('utf-8')
    li = struct.pack('<L', len(i))
    try:  # python 2
        return buffer(zlib.compress(li + lt + i + t + d))
    except NameError:  # python 3
        return zlib.compress(li + lt + i + t + d)