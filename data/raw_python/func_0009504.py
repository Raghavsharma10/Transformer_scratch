def subCell2DGenerator(arr, shape, d01=None, p01=None):
    '''Generator to access evenly sized sub-cells in a 2d array

    Args:
       shape (tuple): number of sub-cells in y,x e.g. (10,15)
       d01 (tuple, optional): cell size in y and x
       p01 (tuple, optional): position of top left edge

    Returns:
        int: 1st index
        int: 2nd index
        array: sub array

    Example:

    >>> a = np.array([[[0,1],[1,2]],[[2,3],[3,4]]])
    >>> gen = subCell2DGenerator(a,(2,2))
    >>> for i,j, sub in gen: print( i,j, sub )
    0 0 [[[0 1]]]
    0 1 [[[1 2]]]
    1 0 [[[2 3]]]
    1 1 [[[3 4]]]
    '''
    for i, j, s0, s1 in subCell2DSlices(arr, shape, d01, p01):
        yield i, j, arr[s0, s1]