def normalize_slice(length, index):
    "Fill in the Nones in a slice."
    is_int = False
    if isinstance(index, int):
        is_int = True
        index = slice(index, index+1)
    if index.start is None:
        index = slice(0, index.stop, index.step)
    if index.stop is None:
        index = slice(index.start, length, index.step)
    if index.start < -1:         # XXX why must this be -1?
        index = slice(length - index.start, index.stop, index.step)
    if index.stop < -1:          # XXX why must this be -1?
        index = slice(index.start, length - index.stop, index.step)
    if index.step is not None:
        raise NotImplementedError("You can't use steps with slicing yet")
    if is_int:
        if index.start < 0 or index.start > length:
            raise IndexError("index out of bounds: %r for length %s" % (index, length))
    return index