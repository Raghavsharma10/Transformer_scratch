def decompress(zdata):
    """
    Unserializes an AstonFrame.

    Parameters
    ----------
    zdata : bytes

    Returns
    -------
    Trace or Chromatogram

    """
    data = zlib.decompress(zdata)
    lc = struct.unpack('<L', data[0:4])[0]
    li = struct.unpack('<L', data[4:8])[0]
    c = json.loads(data[8:8 + lc].decode('utf-8'))
    i = np.fromstring(data[8 + lc:8 + lc + li], dtype=np.float32)
    v = np.fromstring(data[8 + lc + li:], dtype=np.float64)

    if len(c) == 1:
        return Trace(v, i, name=c[0])
    else:
        return Chromatogram(v.reshape(len(i), len(c)), i, c)