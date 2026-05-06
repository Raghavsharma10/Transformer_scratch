def search_full(self, regex, return_string=True, advance_pointer=True):
        """
        Search from the current position.

        If `return_string` is false and a match is found, returns the number of
        characters matched (from the current position *up to* the end of the
        match).

            >>> s = Scanner("test string")
            >>> s.search_full(r' ')
            'test '
            >>> s.pos
            5
            >>> s.search_full(r'i', advance_pointer=False)
            'stri'
            >>> s.pos
            5
            >>> s.search_full(r'i', return_string=False, advance_pointer=False)
            4
            >>> s.pos
            5
        """
        regex = get_regex(regex)
        self.match = regex.search(self.string, self.pos)
        if not self.match:
            return
        start_pos = self.pos
        if advance_pointer:
            self.pos = self.match.end()
        if return_string:
            return self.string[start_pos:self.match.end()]
        return (self.match.end() - start_pos)