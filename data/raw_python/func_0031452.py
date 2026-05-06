def filter(self, **kwargs):
        """
        Add a filter to this C{Reads} instance.

        @param kwargs: Keyword arguments, as accepted by C{ReadFilter}.
        @return: C{self}.
        """
        readFilter = ReadFilter(**kwargs)
        self._filters.append(readFilter.filter)
        return self