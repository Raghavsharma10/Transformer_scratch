def check(self, regex):
        """
        See what :meth:`scan` would return without advancing the pointer.

            >>> s = Scanner("test string")
            >>> s.check('test ')
            'test '
            >>> s.pos
            0
        """
        return self.scan_full(regex, return_string=True, advance_pointer=False)