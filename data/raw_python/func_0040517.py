def _produce_return(self, cursor):
        """ Return the rowcount property from the used cursor.

        Checks the count first, if a count was given.

        :raise ManipulationCheckError: if a row count was set but does not
            match
        """
        rowcount = cursor.rowcount

        # Check the row count?
        if self._rowcount is not None and self._rowcount != rowcount:
            raise ManipulationCheckError(
                "Count was {}, expected {}.".format(rowcount, self._rowcount))

        return rowcount