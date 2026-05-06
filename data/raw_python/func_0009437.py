def minimumLineInArray(arr, relative=False, f=0,
                       refinePosition=True,
                       max_pos=100,
                       return_pos_arr=False,
                       # order=2
                       ):
    '''
    find closest minimum position next to middle line
    relative: return position relative to middle line
    f: relative decrease (0...1) - setting this value close to one will
       discriminate positions further away from the center
    ##order: 2 for cubic refinement
    '''
    s0, s1 = arr.shape[:2]
    if max_pos >= s1:
        x = np.arange(s1)
    else:
        # take fewer positions within 0->(s1-1)
        x = np.rint(np.linspace(0, s1 - 1, min(max_pos, s1))).astype(int)
    res = np.empty((s0, s0), dtype=float)

    _lineSumXY(x, res, arr, f)

    if return_pos_arr:
        return res

    # best integer index
    i, j = np.unravel_index(np.nanargmin(res), res.shape)

    if refinePosition:
        try:
            sub = res[i - 1:i + 2, j - 1:j + 2]
            ii, jj = center_of_mass(sub)
            if not np.isnan(ii):
                i += (ii - 1)
            if not np.isnan(jj):
                j += (jj - 1)
        except TypeError:
            pass


    if not relative:
        return i, j

    hs = (s0 - 1) / 2
    return i - hs, j - hs