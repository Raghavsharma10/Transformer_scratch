def match_length(self):
        """ Find the total length of all words that match between the two sequences."""
        length = 0
        for match in self.get_matching_blocks():
            a, b, size = match
            length += self._text_length(self.a[a:a+size])
        return length