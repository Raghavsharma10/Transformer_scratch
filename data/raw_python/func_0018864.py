def refresh(self) -> None:
        """Update the actual simulation values based on the toy-value pairs.

        Usually, one does not need to call refresh explicitly.  The
        "magic" methods __call__, __setattr__, and __delattr__ invoke
        it automatically, when required.

        Instantiate a 1-dimensional |SeasonalParameter| object:

        >>> from hydpy.core.parametertools import SeasonalParameter
        >>> class Par(SeasonalParameter):
        ...     NDIM = 1
        ...     TYPE = float
        ...     TIME = None
        >>> par = Par(None)
        >>> par.simulationstep = '1d'
        >>> par.shape = (None,)

        When a |SeasonalParameter| object does not contain any toy-value
        pairs yet, the method |SeasonalParameter.refresh| sets all actual
        simulation values to zero:

        >>> par.values = 1.
        >>> par.refresh()
        >>> par.values[0]
        0.0

        When there is only one toy-value pair, its values are relevant
        for all actual simulation values:

        >>> par.toy_1 = 2. # calls refresh automatically
        >>> par.values[0]
        2.0

        Method |SeasonalParameter.refresh| performs a linear interpolation
        for the central time points of each simulation time step.  Hence,
        in the following example, the original values of the toy-value
        pairs do not show up:

        >>> par.toy_12_31 = 4.
        >>> from hydpy import round_
        >>> round_(par.values[0])
        2.00274
        >>> round_(par.values[-2])
        3.99726
        >>> par.values[-1]
        3.0

        If one wants to preserve the original values in this example, one
        would have to set the corresponding toy instances in the middle of
        some simulation step intervals:

        >>> del par.toy_1
        >>> del par.toy_12_31
        >>> par.toy_1_1_12 = 2
        >>> par.toy_12_31_12 = 4.
        >>> par.values[0]
        2.0
        >>> round_(par.values[1])
        2.005479
        >>> round_(par.values[-2])
        3.994521
        >>> par.values[-1]
        4.0
        """
        if not self:
            self.values[:] = 0.
        elif len(self) == 1:
            values = list(self._toy2values.values())[0]
            self.values[:] = self.apply_timefactor(values)
        else:
            for idx, date in enumerate(
                    timetools.TOY.centred_timegrid(self.simulationstep)):
                values = self.interp(date)
                self.values[idx] = self.apply_timefactor(values)