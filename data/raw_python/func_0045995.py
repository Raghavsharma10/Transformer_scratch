def scan(self, regex):
        """
        Match a pattern from the current position.

        If a match is found, advances the scan pointer and returns the matched
        string. Otherwise returns ``None``.

            >>> s = Scanner("test string")
            >>> s.pos
            0
            >>> s.scan(r'foo')
            >>> s.scan(r'bar')
            >>> s.pos
            0
            >>> s.scan(r'test ')
            'test '
            >>> s.pos
            5
        """
        return self.scan_full(regex, return_string=True, advance_pointer=True)