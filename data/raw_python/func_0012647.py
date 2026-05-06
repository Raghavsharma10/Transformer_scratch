def ind2sub(ind, dimensions):
    """
    Calculates subscripts for indices into regularly spaced matrixes.
    """
    # check that the index is within range
    if ind >= np.prod(dimensions):
        raise RuntimeError("ind2sub: index exceeds array size")
    cum_dims = list(dimensions)
    cum_dims.reverse()
    m = 1
    mult = []
    for d in cum_dims:
        m = m*d
        mult.append(m)
    mult.pop()
    mult.reverse()
    mult.append(1)
    indices = []
    for d in mult:
        indices.append((ind/d)+1)
        ind = ind - (ind/d)*d
    return indices