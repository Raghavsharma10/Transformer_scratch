def exists(self, regex):
        """
        See what :meth:`skip_until` would return without advancing the pointer.

            >>> s = Scanner("test string")
            >>> s.exists(' ')
            5
            >>> s.pos
            0

        Returns the number of characters matched if it does exist, or ``None``
        otherwise.
        """
        return self.search_full(regex, return_string=False, advance_pointer=False)