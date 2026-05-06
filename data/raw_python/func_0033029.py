def trace(self, buffer, start, length=1):
        """Reads the points stored in the channel buffer.

        :param buffer: Selects the channel buffer (either 1 or 2).
        :param start: Selects the bin where the reading starts.
        :param length: The number of bins to read.

        .. todo::
           Use binary command TRCB to speed up data transmission.
        """
        # TODO: Do not use transport directly.
        query = 'TRCA? {0}, {1}, {2}'.format(buffer, start, length)
        result = self.transport.ask(query)
        # Result format: "1.0e-004,1.2e-004,". Strip trailing comma then split.
        return (float(f) for f in result.strip(',').split(','))