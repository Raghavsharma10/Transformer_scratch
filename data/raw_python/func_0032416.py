def fit(self, range, function=None):
        """Fits a function to the active display's data trace within a
        specified range of the time window.

        E.g.::

            # Fit's a gaussian to the first 30% of the time window.
            lockin.fit(range=(0, 30), function='gauss')

        :param start: The left limit of the time window in percent.
        :param stop: The right limit of the time window in percent.
        :param function: The function used to fit the data, either 'line',
            'exp', 'gauss' or None, the default. The configured fit function is
            left unchanged if function is None.

        .. note::

            Fitting takes some time. Check the status byte to see when the
            operation is done. A running scan will be paused until the
            fitting is complete.

        .. warning::

            The SR850 will generate an error if the active display trace is not
            stored when the fit command is executed.

        """
        if function is not None:
            self.fit_function = function
        cmd = 'FITT', Integer(min=0, max=100), Integer(min=0, max=100)
        self._write(cmd, start, stop)