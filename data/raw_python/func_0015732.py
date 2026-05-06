def unpack_nullterm_array(array):
    """Takes a null terminated array, copies the values into a list
    and frees each value and the list.
    """

    addrs = cast(array, POINTER(ctypes.c_void_p))
    l = []
    i = 0
    value = array[i]
    while value:
        l.append(value)
        free(addrs[i])
        i += 1
        value = array[i]
    free(addrs)
    return l