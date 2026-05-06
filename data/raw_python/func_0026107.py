def ndarr2bytes(arr, encoding='ascii'):
    """ This is used to ensure that the return value of arr.tostring()
    is actually a *bytes* array in PY3K.  See notes in ndarr2str above.  Even
    though we consider it a bug that numpy's tostring() function returns
    a bytes array in PY3K, there are actually many instances where that is what
    we want - bytes, not unicode.  So we use this function in those
    instances to ensure that when/if this numpy "bug" is "fixed", that
    our calling code still gets bytes where it needs/expects them. """
    # be fast, don't check - just assume 'arr' is a numpy array - the tostring
    # call will fail anyway if not
    retval = arr.tostring()
    # would rather check "if not isinstance(retval, bytes)", but support 2.5.
    if PY3K and isinstance(retval, str):
        # Take note if this ever gets used.  If this ever occurs, it
        # is likely wildly inefficient since numpy.tostring() is now
        # returning unicode and numpy surely has a tobytes() func by now.
        # If so, add a code path to call its tobytes() func at our start.
        return retval.encode(encoding)
    else: # is str==bytes in 2.x
        return retval