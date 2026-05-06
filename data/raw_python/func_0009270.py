def _can_be_double(x):
    """
    Return if the array can be safely converted to double.

    That happens when the dtype is a float with the same size of
    a double or narrower, or when is an integer that can be safely
    converted to double (if the roundtrip conversion works).

    """
    return ((np.issubdtype(x.dtype, np.floating) and
            x.dtype.itemsize <= np.dtype(float).itemsize) or
            (np.issubdtype(x.dtype, np.signedinteger) and
            np.can_cast(x, float)))