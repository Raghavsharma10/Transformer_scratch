def request(self, max_width):
        # type: (int) -> Optional[Tuple[int, Chunk]]
        """Requests a sub-chunk of max_width or shorter. Returns None if no chunks left."""
        if max_width < 1:
            raise ValueError('requires positive integer max_width')

        s = self.chunk.s
        length = len(s)

        if self.internal_offset == len(s):
            return None

        width = 0
        start_offset = i = self.internal_offset
        replacement_char = u' '

        while True:
            w = wcswidth(s[i])

            # If adding a character puts us over the requested width, return what we've got so far
            if width + w > max_width:
                self.internal_offset = i  # does not include ith character
                self.internal_width += width

                # if not adding it us makes us short, this must have been a double-width character
                if width < max_width:
                    assert width + 1 == max_width, 'unicode character width of more than 2!?!'
                    assert w == 2, 'unicode character of width other than 2?'
                    return (width + 1, Chunk(s[start_offset:self.internal_offset] + replacement_char,
                                             atts=self.chunk.atts))
                return (width, Chunk(s[start_offset:self.internal_offset], atts=self.chunk.atts))
            # otherwise add this width
            width += w

            # If one more char would put us over, return whatever we've got
            if i + 1 == length:
                self.internal_offset = i + 1  # beware the fencepost, i is an index not an offset
                self.internal_width += width
                return (width, Chunk(s[start_offset:self.internal_offset], atts=self.chunk.atts))
            # otherwise attempt to add the next character
            i += 1