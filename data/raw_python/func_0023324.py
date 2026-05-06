def iterboxed(self, rows):
        """Iterator that yields each scanline in boxed row flat pixel
        format.  `rows` should be an iterator that yields the bytes of
        each row in turn.
        """

        def asvalues(raw):
            """Convert a row of raw bytes into a flat row.  Result will
            be a freshly allocated object, not shared with
            argument.
            """

            if self.bitdepth == 8:
                return array('B', raw)
            if self.bitdepth == 16:
                raw = tostring(raw)
                return array('H', struct.unpack('!%dH' % (len(raw)//2), raw))
            assert self.bitdepth < 8
            width = self.width
            # Samples per byte
            spb = 8//self.bitdepth
            out = array('B')
            mask = 2**self.bitdepth - 1
            shifts = map(self.bitdepth.__mul__, reversed(range(spb)))
            for o in raw:
                out.extend(map(lambda i: mask&(o>>i), shifts))
            return out[:width]

        return imap(asvalues, rows)