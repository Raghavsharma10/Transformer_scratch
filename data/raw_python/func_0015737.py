def unpack_zeroterm_array(ptr):
    """Converts a zero terminated array to a list and frees each element
    and the list itself.

    If an item is returned all yielded before are invalid.
    """

    assert ptr

    index = 0
    current = ptr[index]
    while current:
        yield current
        free(ffi.cast("gpointer", current))
        index += 1
        current = ptr[index]
    free(ffi.cast("gpointer", ptr))