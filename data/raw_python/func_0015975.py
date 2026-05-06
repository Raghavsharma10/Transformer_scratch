def serialise(structure):
    """
    structure (ctypes.Structure)
        The structure to serialise.

    Returns a ctypes.c_char array.
    Does not copy memory.
    """
    return ctypes.cast(
        ctypes.pointer(structure),
        ctypes.POINTER(ctypes.c_char * ctypes.sizeof(structure)),
    ).contents