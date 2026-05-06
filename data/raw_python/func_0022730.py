def pack_unit(value):
    """Packs float values between [0,1] into 4 unsigned int8

    Returns
    -------
    pack: array
        packed interpolation kernel
    """
    pack = np.zeros(value.shape + (4,), dtype=np.ubyte)
    for i in range(4):
        value, pack[..., i] = np.modf(value * 256.)
    return pack