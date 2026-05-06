def _integrate_mpwrap(ts_and_pks, integrate, fopts):
    """
    Take a zipped timeseries and peaks found in it
    and integrate it to return peaks. Used to allow
    multiprocessing support.
    """
    ts, tpks = ts_and_pks
    pks = integrate(ts, tpks, **fopts)
    # for p in pks:
    #     p.info['mz'] = str(ts.name)
    return pks