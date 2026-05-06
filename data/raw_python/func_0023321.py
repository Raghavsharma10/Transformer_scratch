def array_scanlines_interlace(self, pixels):
        """
        Generator for interlaced scanlines from an array.  `pixels` is
        the full source image in flat row flat pixel format.  The
        generator yields each scanline of the reduced passes in turn, in
        boxed row flat pixel format.
        """

        # http://www.w3.org/TR/PNG/#8InterlaceMethods
        # Array type.
        fmt = 'BH'[self.bitdepth > 8]
        # Value per row
        vpr = self.width * self.planes
        for xstart, ystart, xstep, ystep in _adam7:
            if xstart >= self.width:
                continue
            # Pixels per row (of reduced image)
            ppr = int(math.ceil((self.width-xstart)/float(xstep)))
            # number of values in reduced image row.
            row_len = ppr*self.planes
            for y in range(ystart, self.height, ystep):
                if xstep == 1:
                    offset = y * vpr
                    yield pixels[offset:offset+vpr]
                else:
                    row = array(fmt)
                    # There's no easier way to set the length of an array
                    row.extend(pixels[0:row_len])
                    offset = y * vpr + xstart * self.planes
                    end_offset = (y+1) * vpr
                    skip = self.planes * xstep
                    for i in range(self.planes):
                        row[i::self.planes] = \
                            pixels[offset+i:end_offset:skip]
                    yield row