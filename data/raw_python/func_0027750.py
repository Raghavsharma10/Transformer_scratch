def next(self):
        """
        This method is deprecated, a holdover from when queries were iterators,
        rather than iterables.

        @return: one element of massaged data.
        """
        if self._selfiter is None:
            warnings.warn(
                "Calling 'next' directly on a query is deprecated. "
                "Perhaps you want to use iter(query).next(), or something "
                "more expressive like store.findFirst or store.findOrCreate?",
                DeprecationWarning, stacklevel=2)
            self._selfiter = self.__iter__()
        return self._selfiter.next()