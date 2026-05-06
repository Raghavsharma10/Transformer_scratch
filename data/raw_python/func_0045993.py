def scan_full(self, regex, return_string=True, advance_pointer=True):
        """
        Match from the current position.

        If `return_string` is false and a match is found, returns the number of
        characters matched.

            >>> s = Scanner("test string")
            >>> s.scan_full(r' ')
            >>> s.scan_full(r'test ')
            'test '
            >>> s.pos
            5
            >>> s.scan_full(r'stri', advance_pointer=False)
            'stri'
            >>> s.pos
            5
            >>> s.scan_full(r'stri', return_string=False, advance_pointer=False)
            4
            >>> s.pos
            5
        """
        regex = get_regex(regex)
        self.match = regex.match(self.string, self.pos)
        if not self.match:
            return
        if advance_pointer:
            self.pos = self.match.end()
        if return_string:
            return self.match.group(0)
        return len(self.match.group(0))