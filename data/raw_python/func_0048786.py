def skip(self, n):
        """Skips a specified number of ``bytes`` in the stream.

        arg:    n (cardinal): the number of ``bytes`` to skip
        return: (cardinal) - the actual number of ``bytes`` skipped
        raise:  IllegalState - this stream has been closed or
                ``at_end_of_stream()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._my_data.closed or self.at_end_of_stream():
            raise IllegalState()
        if n is not None:
            self._my_data.seek(n)