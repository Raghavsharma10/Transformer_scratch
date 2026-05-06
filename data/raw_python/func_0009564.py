def calcAspectRatioFromCorners(corners, in_plane=False):
    '''
    simple and better alg. than below
    in_plane -> whether object has no tilt, but only rotation and translation
    '''

    q = corners
    l0 = [q[0, 0], q[0, 1], q[1, 0], q[1, 1]]
    l1 = [q[0, 0], q[0, 1], q[-1, 0], q[-1, 1]]

    l2 = [q[2, 0], q[2, 1], q[3, 0], q[3, 1]]
    l3 = [q[2, 0], q[2, 1], q[1, 0], q[1, 1]]

    a1 = line.length(l0) / line.length(l1)
    a2 = line.length(l2) / line.length(l3)

    if in_plane:
        # take aspect ration from more rectangular corner
        if (abs(0.5 * np.pi - abs(line.angle2(l0, l1)))
                < abs(0.5 * np.pi - abs(line.angle2(l2, l3)))):
            return a1
        else:
            return a2

    return 0.5 * (a1 + a2)