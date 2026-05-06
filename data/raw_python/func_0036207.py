def guess_initc(ts, f, rts=[]):
    """
    ts - An AstonSeries that's being fitted with peaks
    f - The functional form of the peaks (e.g. gaussian)
    rts - peak maxima to fit; each number corresponds to one peak
    """
    def find_side(y, loc=None):
        if loc is None:
            loc = y.argmax()
        ddy = np.diff(np.diff(y))
        lft_loc, rgt_loc = loc - 2, loc + 1
        while rgt_loc >= 0 and rgt_loc < len(ddy):
            if ddy[rgt_loc] < ddy[rgt_loc - 1]:
                break
            rgt_loc += 1
        while lft_loc >= 0 and lft_loc < len(ddy):
            if ddy[lft_loc] < ddy[lft_loc + 1]:
                break
            lft_loc -= 1
        return lft_loc + 1, rgt_loc + 1

    # weight_mom = lambda m, a, w: \
    #   np.sum(w * (a - np.sum(w * a) / np.sum(w)) ** m) / np.sum(w)
    # sig = np.sqrt(weight_mom(2, ts.index, ts.values))  # sigma
    # peak_params['s'] = weight_mom(3, ts.index, ts.values) / sig ** 3
    # peak_params['e'] = weight_mom(4, ts.index, ts.values) / sig ** 4 - 3
    # TODO: better method of calculation of these?
    all_params = []
    for rt in rts:
        peak_params = {'x': rt}  # ts.index[ts.values.argmax()]
        top_idx = np.abs(ts.index - rt).argmin()
        side_idx = find_side(ts.values, top_idx)
        peak_params['h'] = ts.values[top_idx]
        # - min(ts.y[side_idx[0]], ts.y[side_idx[1]])
        peak_params['w'] = ts.index[side_idx[1]] - ts.index[side_idx[0]]
        peak_params['s'] = 1.1
        peak_params['e'] = 1.
        peak_params['a'] = 1.
        all_params.append(peak_params)
    return all_params