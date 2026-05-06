def data_type_to_numpy(datatype, unsigned=False):
    """Convert an ncstream datatype to a numpy one."""
    basic_type = _dtypeLookup[datatype]

    if datatype in (stream.STRING, stream.OPAQUE):
        return np.dtype(basic_type)

    if unsigned:
        basic_type = basic_type.replace('i', 'u')
    return np.dtype('=' + basic_type)