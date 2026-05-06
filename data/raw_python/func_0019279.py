def to_timepoints(self, unit='hours', offset=None):
        """Return an |numpy.ndarray| representing the starting time points
        of the |Timegrid| object.

        The following examples identical with the ones of
        |Timegrid.from_timepoints| but reversed.

        By default, the time points are given in hours:

        >>> from hydpy import Timegrid
        >>> timegrid = Timegrid('2000-01-01', '2000-01-02', '6h')
        >>> timegrid.to_timepoints()
        array([  0.,   6.,  12.,  18.])

        Other time units (`days` or `min`) can be defined (only the first
        character counts):

        >>> timegrid.to_timepoints(unit='d')
        array([ 0.  ,  0.25,  0.5 ,  0.75])

        Additionally, one can pass an `offset` that must be of type |int|
        or an valid |Period| initialization argument:

        >>> timegrid.to_timepoints(offset=24)
        array([ 24.,  30.,  36.,  42.])
        >>> timegrid.to_timepoints(offset='1d')
        array([ 24.,  30.,  36.,  42.])
        >>> timegrid.to_timepoints(unit='day', offset='1d')
        array([ 1.  ,  1.25,  1.5 ,  1.75])
        """
        unit = Period.from_cfunits(unit)
        if offset is None:
            offset = 0.
        else:
            try:
                offset = Period(offset)/unit
            except TypeError:
                offset = offset
        step = self.stepsize/unit
        nmb = len(self)
        variable = numpy.linspace(offset, offset+step*(nmb-1), nmb)
        return variable