def subCell2DSlices(arr, shape, d01=None, p01=None):
    '''Generator to access evenly sized sub-cells in a 2d array

    Args:
       shape (tuple): number of sub-cells in y,x e.g. (10,15)
       d01 (tuple, optional): cell size in y and x
       p01 (tuple, optional): position of top left edge

    Returns:
        int: 1st index
        int: 2nd index
        slice: first dimension
        slice: 1st dimension
    '''
    if p01 is not None:
        yinit, xinit = p01
    else:
        xinit, yinit = 0, 0

    x, y = xinit, yinit
    g0, g1 = shape
    s0, s1 = arr.shape[:2]

    if d01 is not None:
        d0, d1 = d01
    else:
        d0, d1 = s0 / g0, s1 / g1

    y1 = d0 + yinit
    for i in range(g0):
        for j in range(g1):
            x1 = x + d1
            yield (i, j, slice(max(0, _rint(y)),
                               max(0, _rint(y1))),
                   slice(max(0, _rint(x)),
                         max(0, _rint(x1))))
            x = x1
        y = y1
        y1 = y + d0
        x = xinit