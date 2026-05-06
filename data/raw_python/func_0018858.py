def get_timefactor(cls) -> float:
        """Factor to adjust a new value of a time-dependent parameter.

        For a time-dependent parameter, its effective value depends on the
        simulation step size.  Method |Parameter.get_timefactor| returns
        the fraction between the current simulation step size and the
        current parameter step size.

        .. testsetup::

            >>> from hydpy import pub
            >>> del pub.timegrids
            >>> from hydpy.core.parametertools import Parameter
            >>> Parameter.simulationstep.delete()
            Period()

        Method |Parameter.get_timefactor| raises the following error
        when time information is not available:

        >>> from hydpy.core.parametertools import Parameter
        >>> Parameter.get_timefactor()
        Traceback (most recent call last):
        ...
        RuntimeError: To calculate the conversion factor for adapting the \
values of the time-dependent parameters, you need to define both a \
parameter and a simulation time step size first.

        One can define both time step sizes directly:

        >>> _ = Parameter.parameterstep('1d')
        >>> _ = Parameter.simulationstep('6h')
        >>> Parameter.get_timefactor()
        0.25

        As usual, the "global" simulation step size of the |Timegrids|
        object of module |pub| is prefered:

        >>> from hydpy import pub
        >>> pub.timegrids = '2000-01-01', '2001-01-01', '12h'
        >>> Parameter.get_timefactor()
        0.5
        """
        try:
            parfactor = hydpy.pub.timegrids.parfactor
        except RuntimeError:
            if not (cls.parameterstep and cls.simulationstep):
                raise RuntimeError(
                    f'To calculate the conversion factor for adapting '
                    f'the values of the time-dependent parameters, '
                    f'you need to define both a parameter and a simulation '
                    f'time step size first.')
            else:
                date1 = timetools.Date('2000.01.01')
                date2 = date1 + cls.simulationstep
                parfactor = timetools.Timegrids(timetools.Timegrid(
                    date1, date2, cls.simulationstep)).parfactor
        return parfactor(cls.parameterstep)