def Get_RpRs(d, **kwargs):
    '''
    Returns the value of the planet radius over the stellar radius
    for a given depth :py:obj:`d`, given
    the :py:class:`everest.pysyzygy` transit :py:obj:`kwargs`.

    '''
    if ps is None:
            raise Exception("Unable to import `pysyzygy`.")

    def Depth(RpRs, **kwargs):
        return 1 - ps.Transit(RpRs=RpRs, **kwargs)([kwargs.get('t0', 0.)])

    def DiffSq(r):
        return 1.e10 * (d - Depth(r, **kwargs)) ** 2

    return fmin(DiffSq, [np.sqrt(d)], disp=False)