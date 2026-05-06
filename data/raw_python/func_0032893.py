def partial_shape(addr, full_shape):
    """
    Calculate the size of the sub-array represented by `addr`
    """
    def size(x, max):
        if isinstance(x, (int, long, numpy.integer)):
            return None
        elif isinstance(x, slice):
            y = min(max, x.stop or max)  # slice limits can go past the bounds
            return 1 + (y - (x.start or 0) - 1) // (x.step or 1)
        elif isinstance(x, collections.Sized):
            if hasattr(x, 'dtype') and x.dtype == bool:
                return x.sum()
            else:
                return len(x)
        else:
            raise TypeError("Unsupported index type %s" % type(x))

    addr = full_address(addr, full_shape)
    if isinstance(addr, numpy.ndarray) and addr.dtype == bool:
        return (addr.sum(),)
    elif all(isinstance(x, collections.Sized) for x in addr):
        return (len(addr[0]),)
    else:
        shape = [size(x, max) for (x, max) in zip(addr, full_shape)]
        return tuple([x for x in shape if x is not None])