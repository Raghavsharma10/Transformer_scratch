def filter(self, **kwargs):
        """
        Add a filter to this C{readsAlignments}.

        @param kwargs: Keyword arguments, as accepted by
            C{ReadsAlignmentsFilter}.
        @return: C{self}
        """
        self._filters.append(ReadsAlignmentsFilter(**kwargs).filter)
        return self