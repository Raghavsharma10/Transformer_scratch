def sortCorners(corners):
    '''
    sort the corners of a given quadrilateral of the type
    corners : [ [xi,yi],... ]

    to an anti-clockwise order starting with the bottom left corner

    or (if plotted as image where y increases to the bottom):
    clockwise, starting top left
    '''
    corners = np.asarray(corners)
    # bring edges in order:
    corners2 = corners[ConvexHull(corners).vertices]
   
    if len(corners2) == 3:
        # sometimes ConvexHull one point is missing because it is
        # within the hull triangle
        # find the right position of set corner as the minimum perimeter
        # built with that point as different indices
        for c in corners:
            if c not in corners2:
                break
        perimeter = []
        for n in range(0, 4):
            corners3 = np.insert(corners2, n, c, axis=0)
            perimeter.append(
                np.linalg.norm(
                    np.diff(
                        corners3,
                        axis=0),
                    axis=1).sum())
        n = np.argmin(perimeter)
        corners2 = np.insert(corners2, n, c, axis=0)

    # find the edge with the right angle to the quad middle:
    mn = corners2.mean(axis=0)
    d = (corners2 - mn)
    ascent = np.arctan2(d[:, 1], d[:, 0])
    bl = np.abs(BL_ANGLE + ascent).argmin()
    # build a index list starting with bl:
    i = list(range(bl, 4))
    i.extend(list(range(0, bl)))
    return corners2[i]