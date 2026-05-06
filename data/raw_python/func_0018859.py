def apply_timefactor(cls, values):
        """Change and return the given value(s) in accordance with
        |Parameter.get_timefactor| and the type of time-dependence
        of the actual parameter subclass.

        .. testsetup::

            >>> from hydpy import pub
            >>> del pub.timegrids

        For the same conversion factor returned by method
        |Parameter.get_timefactor|, method |Parameter.apply_timefactor|
        behaves differently depending on the `TIME` attribute of the
        respective |Parameter| subclass.  We first prepare a parameter
        test class and define both the parameter and simulation step size:

        >>> from hydpy.core.parametertools import Parameter
        >>> class Par(Parameter):
        ...     TIME = None
        >>> Par.parameterstep = '1d'
        >>> Par.simulationstep = '6h'

        |None| means the value(s) of the parameter are not time-dependent
        (e.g. maximum storage capacity).  Hence, |Parameter.apply_timefactor|
        returns the original value(s):

        >>> Par.apply_timefactor(4.0)
        4.0

        |True| means the effective parameter value is proportional to
        the simulation step size (e.g. travel time). Hence,
        |Parameter.apply_timefactor| returns a reduced value in the
        next example (where the simulation step size is smaller than
        the parameter step size):

        >>> Par.TIME = True
        >>> Par.apply_timefactor(4.0)
        1.0

        |False| means the effective parameter value is inversely
        proportional to the simulation step size (e.g. storage
        coefficient). Hence, |Parameter.apply_timefactor| returns
        an increased value in the next example:

        >>> Par.TIME = False
        >>> Par.apply_timefactor(4.0)
        16.0
        """
        if cls.TIME is True:
            return values * cls.get_timefactor()
        if cls.TIME is False:
            return values / cls.get_timefactor()
        return values