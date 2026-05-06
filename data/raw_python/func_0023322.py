def undo_filter(self, filter_type, scanline, previous):
        """Undo the filter for a scanline.  `scanline` is a sequence of
        bytes that does not include the initial filter type byte.
        `previous` is decoded previous scanline (for straightlaced
        images this is the previous pixel row, but for interlaced
        images, it is the previous scanline in the reduced image, which
        in general is not the previous pixel row in the final image).
        When there is no previous scanline (the first row of a
        straightlaced image, or the first row in one of the passes in an
        interlaced image), then this argument should be ``None``.

        The scanline will have the effects of filtering removed, and the
        result will be returned as a fresh sequence of bytes.
        """

        # :todo: Would it be better to update scanline in place?
        # Yes, with the Cython extension making the undo_filter fast,
        # updating scanline inplace makes the code 3 times faster
        # (reading 50 images of 800x800 went from 40s to 16s)
        result = scanline

        if filter_type == 0:
            return result

        if filter_type not in (1,2,3,4):
            raise FormatError('Invalid PNG Filter Type.'
              '  See http://www.w3.org/TR/2003/REC-PNG-20031110/#9Filters .')

        # Filter unit.  The stride from one pixel to the corresponding
        # byte from the previous pixel.  Normally this is the pixel
        # size in bytes, but when this is smaller than 1, the previous
        # byte is used instead.
        fu = max(1, self.psize)

        # For the first line of a pass, synthesize a dummy previous
        # line.  An alternative approach would be to observe that on the
        # first line 'up' is the same as 'null', 'paeth' is the same
        # as 'sub', with only 'average' requiring any special case.
        if not previous:
            previous = array('B', [0]*len(scanline))

        def sub():
            """Undo sub filter."""

            ai = 0
            # Loop starts at index fu.  Observe that the initial part
            # of the result is already filled in correctly with
            # scanline.
            for i in range(fu, len(result)):
                x = scanline[i]
                a = result[ai]
                result[i] = (x + a) & 0xff
                ai += 1

        def up():
            """Undo up filter."""

            for i in range(len(result)):
                x = scanline[i]
                b = previous[i]
                result[i] = (x + b) & 0xff

        def average():
            """Undo average filter."""

            ai = -fu
            for i in range(len(result)):
                x = scanline[i]
                if ai < 0:
                    a = 0
                else:
                    a = result[ai]
                b = previous[i]
                result[i] = (x + ((a + b) >> 1)) & 0xff
                ai += 1

        def paeth():
            """Undo Paeth filter."""

            # Also used for ci.
            ai = -fu
            for i in range(len(result)):
                x = scanline[i]
                if ai < 0:
                    a = c = 0
                else:
                    a = result[ai]
                    c = previous[ai]
                b = previous[i]
                p = a + b - c
                pa = abs(p - a)
                pb = abs(p - b)
                pc = abs(p - c)
                if pa <= pb and pa <= pc:
                    pr = a
                elif pb <= pc:
                    pr = b
                else:
                    pr = c
                result[i] = (x + pr) & 0xff
                ai += 1

        # Call appropriate filter algorithm.  Note that 0 has already
        # been dealt with.
        (None,
         pngfilters.undo_filter_sub,
         pngfilters.undo_filter_up,
         pngfilters.undo_filter_average,
         pngfilters.undo_filter_paeth)[filter_type](fu, scanline, previous, result)
        return result