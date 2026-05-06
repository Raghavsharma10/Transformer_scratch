def drop_integrate(ts, peak_list):
    """
    Resolves overlap by breaking at the minimum value.
    """
    peaks = []
    for _, pks in _get_windows(peak_list):
        temp_pks = []
        pks = sorted(pks, key=lambda p: p['t0'])
        if 'y0' in pks[0] and 'y1' in pks[-1]:
            y0, y1 = pks[0]['y0'], pks[-1]['y1']
        else:
            y0 = ts.get_point(pks[0]['t0'])
            y1 = ts.get_point(pks[-1]['t1'])
        ys = np.array([y0, y1])
        xs = np.array([pks[0]['t0'], pks[-1]['t1']])

        # go through list of peaks to make sure there's no overlap
        for hints in pks:
            t0, t1 = hints['t0'], hints['t1']

            # figure out the y values (using a linear baseline)
            hints['y0'] = np.interp(t0, xs, ys)
            hints['y1'] = np.interp(t1, xs, ys)

            # if this peak totally overlaps with an existing one, don't add
            if sum(1 for p in temp_pks if t1 <= p['t1']) > 0:
                continue
            overlap_pks = [p for p in temp_pks if t0 <= p['t1']]
            if len(overlap_pks) > 0:
                # find the last of the overlapping peaks
                overlap_pk = max(overlap_pks, key=lambda p: p['t0'])
                # get the section of trace and find the lowest point
                over_ts = ts.twin((t0, overlap_pk['t1']))
                min_t = over_ts.index[over_ts.values.argmin()]

                # delete the existing overlaping peak
                for i, p in enumerate(temp_pks):
                    if p == overlap_pk:
                        del temp_pks[i]
                        break

                # interpolate a new y value
                y_val = np.interp(min_t, xs, ys)
                overlap_pk['y1'] = y_val
                hints['y0'] = y_val

                # add the old and new peak in
                overlap_pk['t1'] = min_t
                temp_pks.append(overlap_pk)
                hints['t0'], hints['t1'] = min_t, t1
                temp_pks.append(hints)
            else:
                hints['t0'], hints['t1'] = t0, t1
                temp_pks.append(hints)

        # none of our peaks should overlap, so we can just use
        # simple_integrate now
        peaks += simple_integrate(ts, temp_pks, intname='drop')
    return peaks