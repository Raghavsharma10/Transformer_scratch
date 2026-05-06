def _queryContainer(self, store):
        """
        Generate and cache the subselect SQL and its arguments.  Return the
        subselect SQL.
        """
        if self._subselectSQL is None:
            sql, args = self.container._sqlAndArgs('SELECT',
                                                   self.container._queryTarget)
            self._subselectSQL, self._subselectArgs = sql, args
        return self._subselectSQL