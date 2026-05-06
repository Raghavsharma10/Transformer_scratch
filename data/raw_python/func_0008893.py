def _tarboton_slopes_directions(data, dX, dY, facets, ang_adj):
    """
    Calculate the slopes and directions based on the 8 sections from
    Tarboton http://www.neng.usu.edu/cee/faculty/dtarb/96wr03137.pdf
    """
    shp = np.array(data.shape) - 1


    direction = np.full(data.shape, FLAT_ID_INT, 'float64')
    mag = np.full(data.shape, FLAT_ID_INT, 'float64')

    slc0 = [slice(1, -1), slice(1, -1)]
    for ind in xrange(8):

        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(1 + e1[0], shp[0] + e1[0]),
                slice(1 + e1[1], shp[1] + e1[1])]
        slc2 = [slice(1 + e2[0], shp[0] + e2[0]),
                slice(1 + e2[1], shp[1] + e2[1])]

        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp)
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)

    # %%Now do the edges
    # if the edge is lower than the interior, we need to copy the value
    # from the interior (as an approximation)
    ids1 = (direction[:, 1] > np.pi / 2) \
        & (direction[:, 1] < 3 * np.pi / 2)
    direction[ids1, 0] = direction[ids1, 1]
    mag[ids1, 0] = mag[ids1, 1]
    ids1 = (direction[:, -2] < np.pi / 2) \
        | (direction[:, -2] > 3 * np.pi / 2)
    direction[ids1, -1] = direction[ids1, -2]
    mag[ids1, -1] = mag[ids1, -2]
    ids1 = (direction[1, :] > 0) & (direction[1, :] < np.pi)
    direction[0, ids1] = direction[1, ids1]
    mag[0, ids1] = mag[1, ids1]
    ids1 = (direction[-2, :] > np.pi) & (direction[-2, :] < 2 * np.pi)
    direction[-1, ids1] = direction[-2, ids1]
    mag[-1, ids1] = mag[-2, ids1]

    # Now update the edges in case they are higher than the interior (i.e.
    # look at the downstream angle)

    # left edge
    slc0 = [slice(1, -1), slice(0, 1)]
    for ind in [0, 1, 6, 7]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(1 + e1[0], shp[0] + e1[0]), slice(e1[1], 1 + e1[1])]
        slc2 = [slice(1 + e2[0], shp[0] + e2[0]), slice(e2[1], 1 + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp)
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # right edge
    slc0 = [slice(1, -1), slice(-1, None)]
    for ind in [2, 3, 4, 5]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(1 + e1[0], shp[0] + e1[0]),
                slice(shp[1] + e1[1], shp[1] + 1 + e1[1])]
        slc2 = [slice(1 + e2[0], shp[0] + e2[0]),
                slice(shp[1] + e2[1], shp[1] + 1 + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp)
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # top edge
    slc0 = [slice(0, 1), slice(1, -1)]
    for ind in [4, 5, 6, 7]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(e1[0], 1 + e1[0]), slice(1 + e1[1], shp[1] + e1[1])]
        slc2 = [slice(e2[0], 1 + e2[0]), slice(1 + e2[1], shp[1] + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'top')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # bottom edge
    slc0 = [slice(-1, None), slice(1, -1)]
    for ind in [0, 1, 2, 3]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(shp[0] + e1[0], shp[0] + 1 + e1[0]),
                slice(1 + e1[1], shp[1] + e1[1])]
        slc2 = [slice(shp[0] + e2[0], shp[0] + 1 + e2[0]),
                slice(1 + e2[1], shp[1] + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'bot')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # top-left corner
    slc0 = [slice(0, 1), slice(0, 1)]
    for ind in [6, 7]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(e1[0], 1 + e1[0]), slice(e1[1], 1 + e1[1])]
        slc2 = [slice(e2[0], 1 + e2[0]), slice(e2[1], 1 + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'top')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # top-right corner
    slc0 = [slice(0, 1), slice(-1, None)]
    for ind in [4, 5]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(e1[0], 1 + e1[0]),
                slice(shp[1] + e1[1], shp[1] + 1 + e1[1])]
        slc2 = [slice(e2[0], 1 + e2[0]),
                slice(shp[1] + e2[1], shp[1] + 1 + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'top')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # bottom-left corner
    slc0 = [slice(-1, None), slice(0, 1)]
    for ind in [0, 1]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(shp[0] + e1[0], shp[0] + 1 + e1[0]),
                slice(e1[1], 1 + e1[1])]
        slc2 = [slice(shp[0] + e2[0], shp[0] + 1 + e2[0]),
                slice(e2[1], 1 + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'bot')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # bottom-right corner
    slc0 = [slice(-1, None), slice(-1, None)]
    for ind in [3, 4]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = [slice(shp[0] + e1[0], shp[0] + 1 + e1[0]),
                slice(shp[1] + e1[1], shp[1] + 1 + e1[1])]
        slc2 = [slice(shp[0] + e2[0], shp[0] + 1 + e2[0]),
                slice(shp[1] + e2[1], shp[1] + 1 + e2[1])]
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'bot')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)

    mag[mag > 0] = np.sqrt(mag[mag > 0])

    return mag, direction