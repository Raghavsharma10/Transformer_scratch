def interpolate2dUnstructuredIDW(x, y, v, grid, power=2):
    '''
    x,y,v --> 1d numpy.array
    grid --> 2d numpy.array

    fast if number of given values is small relative to grid resolution
    '''
    n = len(v)
    gx = grid.shape[0]
    gy = grid.shape[1]
    for i in range(gx):
        for j in range(gy):
            overPx = False  # if pixel position == point position
            sumWi = 0.0
            value = 0.0

            for k in range(n):
                xx = x[k]
                yy = y[k]
                vv = v[k]
                if xx == i and yy == j:
                    grid[i, j] = vv
                    overPx = True
                    break
                # weight from inverse distance:
                wi = 1 / ((xx - i)**2 + (yy - j)**2)**(0.5 * power)
                sumWi += wi
                value += wi * vv
            if not overPx:
                grid[i, j] = value / sumWi
    return grid