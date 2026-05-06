def simple_peak_find(s, init_slope=500, start_slope=500, end_slope=200,
                     min_peak_height=50, max_peak_width=1.5):
    """
    Given a Series, return a list of tuples indicating when
    peaks start and stop and what their baseline is.
    [(t_start, t_end, hints) ...]
    """
    point_gap = 10

    def slid_win(itr, size=2):
        """Returns a sliding window of size 'size' along itr."""
        itr, buf = iter(itr), []
        for _ in range(size):
            buf += [next(itr)]
        for l in itr:
            yield buf
            buf = buf[1:] + [l]
        yield buf

    # TODO: check these smoothing defaults
    y, t = s.values, s.index.astype(float)
    smooth_y = movingaverage(y, 9)
    dxdt = np.gradient(smooth_y) / np.gradient(t)
    # dxdt = -savitzkygolay(ts, 5, 3, deriv=1).y / np.gradient(t)

    init_slopes = np.arange(len(dxdt))[dxdt > init_slope]
    if len(init_slopes) == 0:
        return []
    # get the first points of any "runs" as a peak start
    # runs can have a gap of up to 10 points in them
    peak_sts = [init_slopes[0]]
    peak_sts += [j for i, j in slid_win(init_slopes, 2) if j - i > 10]
    peak_sts.sort()

    en_slopes = np.arange(len(dxdt))[dxdt < -end_slope]
    if len(en_slopes) == 0:
        return []
    # filter out any lone points farther than 10 away from their neighbors
    en_slopes = [en_slopes[0]]
    en_slopes += [i[1] for i in slid_win(en_slopes, 3)
                  if i[1] - i[0] < point_gap or i[2] - i[1] < point_gap]
    en_slopes += [en_slopes[-1]]
    # get the last points of any "runs" as a peak end
    peak_ens = [j for i, j in slid_win(en_slopes[::-1], 2)
                if i - j > point_gap] + [en_slopes[-1]]
    peak_ens.sort()
    # avals = np.arange(len(t))[np.abs(t - 0.675) < 0.25]
    # print([i for i in en_slopes if i in avals])
    # print([(t[i], i) for i in peak_ens if i in avals])

    peak_list = []
    pk2 = 0
    for pk in peak_sts:
        # don't allow overlapping peaks
        if pk < pk2:
            continue

        # track backwards to find the true start
        while dxdt[pk] > start_slope and pk > 0:
            pk -= 1

        # now find where the peak ends
        dist_to_end = np.array(peak_ens) - pk
        pos_end = pk + dist_to_end[dist_to_end > 0]
        for pk2 in pos_end:
            if (y[pk2] - y[pk]) / (t[pk2] - t[pk]) > start_slope:
                # if the baseline beneath the peak is too large, let's
                # keep going to the next dip
                peak_list.append({'t0': t[pk], 't1': t[pk2]})
                pk = pk2
            elif t[pk2] - t[pk] > max_peak_width:
                # make sure that peak is short enough
                pk2 = pk + np.abs(t[pk:] - t[pk] - max_peak_width).argmin()
                break
            else:
                break
        else:
            # if no end point is found, the end point
            # is the end of the timeseries
            pk2 = len(t) - 1

        if pk == pk2:
            continue
        pk_hgt = max(y[pk:pk2]) - min(y[pk:pk2])
        if pk_hgt < min_peak_height:
            continue
        peak_list.append({'t0': t[pk], 't1': t[pk2]})
    return peak_list