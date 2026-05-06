def scan_until(self, regex):
        """
        Search for a pattern from the current position.

        If a match is found, advances the scan pointer and returns the matched
        string, from the current position *up to* the end of the match.
        Otherwise returns ``None``.

            >>> s = Scanner("test string")
            >>> s.pos
            0
            >>> s.scan_until(r'foo')
            >>> s.scan_until(r'bar')
            >>> s.pos
            0
            >>> s.scan_until(r' ')
            'test '
            >>> s.pos
            5
        """
        return self.search_full(regex, return_string=True, advance_pointer=True)