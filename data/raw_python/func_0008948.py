def grow_slice(slc, size):
    """
    Grow a slice object by 1 in each direction without overreaching the list.

    Parameters
    ----------
    slc: slice
        slice object to grow
    size: int
        list length

    Returns
    -------
    slc: slice
       extended slice 

    """

    return slice(max(0, slc.start-1), min(size, slc.stop+1))