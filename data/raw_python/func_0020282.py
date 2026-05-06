def Transit(time, t0=0., dur=0.1, per=3.56789, depth=0.001, **kwargs):
    '''
    A `Mandel-Agol <http://adsabs.harvard.edu/abs/2002ApJ...580L.171M>`_
    transit model, but with the depth and the duration as primary
    input variables.

    :param numpy.ndarray time: The time array
    :param float t0: The time of first transit in units of \
           :py:obj:`BJD` - 2454833.
    :param float dur: The transit duration in days. Don't go too crazy on \
           this one -- very small or very large values will break the \
           inverter. Default 0.1
    :param float per: The orbital period in days. Default 3.56789
    :param float depth: The fractional transit depth. Default 0.001
    :param dict kwargs: Any additional keyword arguments, passed directly \
           to :py:func:`pysyzygy.Transit`
    :returns tmod: The transit model evaluated at the same times as the \
                   :py:obj:`time` array

    '''
    if ps is None:
        raise Exception("Unable to import `pysyzygy`.")

    # Note that rhos can affect RpRs, so we should really do this iteratively,
    # but the effect is pretty negligible!
    RpRs = Get_RpRs(depth, t0=t0, per=per, **kwargs)
    rhos = Get_rhos(dur, t0=t0, per=per, **kwargs)
    return ps.Transit(t0=t0, per=per, RpRs=RpRs, rhos=rhos, **kwargs)(time)