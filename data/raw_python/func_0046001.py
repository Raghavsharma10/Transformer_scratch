def check_until(self, regex):
        """
        See what :meth:`scan_until` would return without advancing the pointer.

            >>> s = Scanner("test string")
            >>> s.check_until(' ')
            'test '
            >>> s.pos
            0
        """
        return self.search_full(regex, return_string=True, advance_pointer=False)