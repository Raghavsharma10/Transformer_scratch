def closestConnectedDistance(target, walls=None,
                             max_len_border_line=500,
                             max_n_path=100,
                             concentrate_every_n_pixel=1):
    '''
    returns an array with contains the closest distance from every pixel
    the next position where target == 1

    [walls] binary 2darray - e.g. walls in a labyrinth that have to be surrounded in order to get to the target

    [target] binary 2darray - positions given by 1

    [concentrate_every_n_pixel] often the distance of neighbour pixels is similar
            to speed up calculation set this value to e.g. 3 to calculate only
            the distance for every 3. pixel and interpolate in between

        recommended are values up to 3-5

    [max_len_border_line]
    this function calculates distances travelled using region growth
    e.g.

    0123
    1123
    2223
    3333

    the last steps (e.g. for all steps 3 border_line=7) are stored in an array of limited
    length defined in 'max_len_border_line'

    [max_n_path]
    how many paths are possible between every pixel and the target
    only needed if fast==False
    '''
    c = concentrate_every_n_pixel
    assert c >= 1
    if walls is None:
        walls = np.zeros_like(target, dtype=bool)
    s = target.shape
    dt = np.uint16
    if max(target.shape) < 200:
        dt = np.uint8
    out = np.zeros((s[0] // c, s[1] // c), dtype=dt)
    # temporary arrays:
    growth = np.zeros_like(target, dtype=dt)
    res = np.empty(shape=3, dtype=dt)

    steps = np.empty(shape=(max_len_border_line, 2), dtype=dt)
    new_steps = np.empty(shape=(max_len_border_line, 2), dtype=dt)

    # run calculation:
    _calc(growth, out, walls, target, steps, new_steps,
          res, concentrate_every_n_pixel)

    if c > 1:
        # if concentrate_every_n_pixel > 1
        # the resized output array
        # will have wrong values close to the wall
        # therefore substitute all wall value (-1)
        # with an average of their closest neighbours
        interpolate2dStructuredIDW(out, out == 0)
        out = cv2.resize(out, s[::-1])
        out[walls] = 0
    return out