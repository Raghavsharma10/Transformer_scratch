def from_timepoints(cls, timepoints, refdate, unit='hours'):
        """Return a |Timegrid| object representing the given starting
        `timepoints` in relation to the given `refdate`.

        The following examples identical with the ones of
        |Timegrid.to_timepoints| but reversed.

        At least two given time points must be increasing and
        equidistant.  By default, they are assumed in hours since
        the given reference date:

        >>> from hydpy import Timegrid
        >>> Timegrid.from_timepoints(
        ...     [0.0, 6.0, 12.0, 18.0], '01.01.2000')
        Timegrid('01.01.2000 00:00:00',
                 '02.01.2000 00:00:00',
                 '6h')
        >>> Timegrid.from_timepoints(
        ...     [24.0, 30.0, 36.0, 42.0], '1999-12-31')
        Timegrid('2000-01-01 00:00:00',
                 '2000-01-02 00:00:00',
                 '6h')

        Other time units (`days` or `min`) must be passed explicitely
        (only the first character counts):

        >>> Timegrid.from_timepoints(
        ...     [0.0, 0.25, 0.5, 0.75], '01.01.2000', unit='d')
        Timegrid('01.01.2000 00:00:00',
                 '02.01.2000 00:00:00',
                 '6h')
        >>> Timegrid.from_timepoints(
        ...     [1.0, 1.25, 1.5, 1.75], '1999-12-31', unit='day')
        Timegrid('2000-01-01 00:00:00',
                 '2000-01-02 00:00:00',
                 '6h')
        """
        refdate = Date(refdate)
        unit = Period.from_cfunits(unit)
        delta = timepoints[1]-timepoints[0]
        firstdate = refdate+timepoints[0]*unit
        lastdate = refdate+(timepoints[-1]+delta)*unit
        stepsize = (lastdate-firstdate)/len(timepoints)
        return cls(firstdate, lastdate, stepsize)