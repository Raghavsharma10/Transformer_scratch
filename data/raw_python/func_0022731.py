def pack_ieee(value):
    """Packs float ieee binary representation into 4 unsigned int8

    Returns
    -------
    pack: array
        packed interpolation kernel
    """
    return np.fromstring(value.tostring(),
                         np.ubyte).reshape((value.shape + (4,)))