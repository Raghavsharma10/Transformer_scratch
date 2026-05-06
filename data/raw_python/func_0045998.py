def skip(self, regex):
        """
        Like :meth:`scan`, but return the number of characters matched.

            >>> s = Scanner("test string")
            >>> s.skip('test ')
            5
        """
        return self.scan_full(regex, return_string=False, advance_pointer=True)