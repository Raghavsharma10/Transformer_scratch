def skip_until(self, regex):
        """
        Like :meth:`scan_until`, but return the number of characters matched.

            >>> s = Scanner("test string")
            >>> s.skip_until(' ')
            5
        """
        return self.search_full(regex, return_string=False, advance_pointer=True)