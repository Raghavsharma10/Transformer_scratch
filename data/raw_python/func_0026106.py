def ndarr2str(arr, encoding='ascii'):
    """ This is used to ensure that the return value of arr.tostring()
    is actually a string.  This will prevent lots of if-checks in calling
    code.  As of numpy v1.6.1 (in Python 3.2.3), the tostring() function
    still returns type 'bytes', not 'str' as it advertises. """
    # be fast, don't check - just assume 'arr' is a numpy array - the tostring
    # call will fail anyway if not
    retval = arr.tostring()
    # would rather check "if isinstance(retval, bytes)", but support 2.5.
    # could rm the if PY3K check, but it makes this faster on 2.x.
    if PY3K and not isinstance(retval, str):
        return retval.decode(encoding)
    else: # is str
        return retval