def close(self):
        """Closes the cursor.
        No further operations are allowed once the cursor is closed.

        If the cursor is used in a ``with`` statement, this method will
        be automatically called at the end of the ``with`` block.
        """
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        if self._id is not None:
            self._connection._client.close_statement(self._connection._id, self._id)
            self._id = None
        self._signature = None
        self._column_data_types = []
        self._frame = None
        self._pos = None
        self._closed = True