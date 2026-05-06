def close(self):
        """Close this ResultProxy.
        Closes the underlying DBAPI cursor corresponding to the execution.
        Note that any data cached within this ResultProxy is still available.
        For some types of results, this may include buffered rows.
        If this ResultProxy was generated from an implicit execution,
        the underlying Connection will also be closed (returns the
        underlying DBAPI connection to the connection pool.)
        This method is called automatically when:
        * all result rows are exhausted using the fetchXXX() methods.
        * cursor.description is None.
        """

        if not self._closed:
            self._closed = True
            yield from self._cursor.close()
            # allow consistent errors
            self._cursor = None
            self._weak = None
        else:
            # pragma: no cover
            pass