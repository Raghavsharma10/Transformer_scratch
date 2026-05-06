def _get_d1_d2(dX, dY, ind, e1, e2, shp, topbot=None):
    """
    This finds the distances along the patch (within the eight neighboring
    pixels around a central pixel) given the difference in x and y coordinates
    of the real image. This is the function that allows real coordinates to be
    used when calculating the magnitude and directions of slopes.
    """
    if topbot == None:
        if ind in [0, 3, 4, 7]:
            d1 = dX[slice((e2[0] + 1) / 2, shp[0] + (e2[0] - 1) / 2)]
            d2 = dY[slice((e2[0] + 1) / 2, shp[0] + (e2[0] - 1) / 2)]
            if d1.size == 0:
                d1 = np.array([dX[0]])
                d2 = np.array([dY[0]])
        else:
            d2 = dX[slice((e1[0] + 1) / 2, shp[0] + (e1[0] - 1) / 2)]
            d1 = dY[slice((e1[0] + 1) / 2, shp[0] + (e1[0] - 1) / 2)]
            if d1.size == 0:
                d2 = dX[0]
                d1 = dY[0]
    elif topbot == 'top':
        if ind in [0, 3, 4, 7]:
            d1, d2 = dX[0], dY[0]
        else:
            d2, d1 = dX[0], dY[0]
    elif topbot == 'bot':
        if ind in [0, 3, 4, 7]:
            d1, d2 = dX[-1], dY[-1]
        else:
            d2, d1 = dX[-1], dY[-1]

    theta = np.arctan2(d2, d1)

    return d1.reshape(d1.size, 1), d2.reshape(d2.size, 1), theta.reshape(theta.size, 1)