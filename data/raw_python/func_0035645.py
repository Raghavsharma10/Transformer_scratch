def simple_integrate(ts, peak_list, base_ts=None, intname='simple'):
    """
    Integrate each peak naively; without regard to overlap.

    This is used as the terminal step by most of the other integrators.
    """
    peaks = []
    for hints in peak_list:
        t0, t1 = hints['t0'], hints['t1']
        hints['int'] = intname
        pk_ts = ts.twin((t0, t1))
        if base_ts is None:
            # make a two point baseline
            base = Trace([hints.get('y0', pk_ts[0]),
                          hints.get('y1', pk_ts[-1])],
                         [t0, t1], name=ts.name)
        else:
            base = base_ts.twin((t0, t1))
        peaks.append(PeakComponent(hints, pk_ts, base))
    return peaks