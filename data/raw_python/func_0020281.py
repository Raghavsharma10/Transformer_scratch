def Get_rhos(dur, **kwargs):
    '''
    Returns the value of the stellar density for a given transit
    duration :py:obj:`dur`, given
    the :py:class:`everest.pysyzygy` transit :py:obj:`kwargs`.

    '''
    if ps is None:
            raise Exception("Unable to import `pysyzygy`.")

    assert dur >= 0.01 and dur <= 0.5, "Invalid value for the duration."

    def Dur(rhos, **kwargs):
        t0 = kwargs.get('t0', 0.)
        time = np.linspace(t0 - 0.5, t0 + 0.5, 1000)
        try:
            t = time[np.where(ps.Transit(rhos=rhos, **kwargs)(time) < 1)]
        except:
            return 0.
        return t[-1] - t[0]

    def DiffSq(rhos):
        return (dur - Dur(rhos, **kwargs)) ** 2

    return fmin(DiffSq, [0.2], disp=False)