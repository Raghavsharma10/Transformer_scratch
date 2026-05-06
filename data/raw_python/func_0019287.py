def centred_timegrid(cls, simulationstep):
        """Return a |Timegrid| object defining the central time points
        of the year 2000 for the given simulation step.

        >>> from hydpy.core.timetools import TOY
        >>> TOY.centred_timegrid('1d')
        Timegrid('2000-01-01 12:00:00',
                 '2001-01-01 12:00:00',
                 '1d')
        """
        simulationstep = Period(simulationstep)
        return Timegrid(
            cls._STARTDATE+simulationstep/2,
            cls._ENDDATE+simulationstep/2,
            simulationstep)