def cursor(self, cursor_factory=None):
        """Creates a new cursor.

        :param cursor_factory:
            This argument can be used to create non-standard cursors.
            The class returned must be a subclass of
            :class:`~phoenixdb.cursor.Cursor` (for example :class:`~phoenixdb.cursor.DictCursor`).
            A default factory for the connection can also be specified using the
            :attr:`cursor_factory` attribute.

        :returns:
            A :class:`~phoenixdb.cursor.Cursor` object.
        """
        if self._closed:
            raise ProgrammingError('the connection is already closed')
        cursor = (cursor_factory or self.cursor_factory)(self)
        self._cursors.append(weakref.ref(cursor, self._cursors.remove))
        return cursor