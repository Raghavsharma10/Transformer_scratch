def calculate_statistics(self, start, stop):
        """Starts the statistics calculation.

        :param start: The left limit of the time window in percent.
        :param stop: The right limit of the time window in percent.

        .. note::

            The calculation takes some time. Check the status byte to see when
            the operation is done. A running scan will be paused until the
            operation is complete.

        .. warning::

            The SR850 will generate an error if the active display trace is not
            stored when the command is executed.

        """
        cmd = 'STAT', Integer, Integer
        self._write(cmd, start, stop)