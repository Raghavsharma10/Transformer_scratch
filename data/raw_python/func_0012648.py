def sub2ind(indices, dimensions):
    """
    An exemplary sub2ind implementation to create randomization 
    scripts. 

    This function calculates indices from subscripts into regularly spaced
    matrixes.
    """
    # check that none of the indices exceeds the size of the array
    if any([i > j for i, j in zip(indices, dimensions)]):
        raise RuntimeError("sub2ind:an index exceeds its dimension's size")
    dims = list(dimensions)
    dims.append(1)
    dims.remove(dims[0])
    dims.reverse()
    ind  = list(indices)
    ind.reverse()
    idx = 0
    mult = 1
    for (cnt, dim) in zip(ind, dims):
        mult = dim*mult
        idx = idx + (cnt-1)*mult    
    return idx