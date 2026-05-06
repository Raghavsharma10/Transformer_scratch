def pairwise(fun, v):
    """
    >>> pairwise(operator.sub, [4,3,2,1,-10])
    [1, 1, 1, 11]
    >>> import numpy
    >>> pairwise(numpy.subtract, numpy.array([4,3,2,1,-10]))
    array([ 1,  1,  1, 11])
    """
    if not hasattr(v, 'shape'):
        return list(ipairwise(fun,v))
    else:
        return fun(v[:-1],v[1:])