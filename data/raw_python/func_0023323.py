def deinterlace(self, raw):
        """
        Read raw pixel data, undo filters, deinterlace, and flatten.
        Return in flat row flat pixel format.
        """

        # Values per row (of the target image)
        vpr = self.width * self.planes

        # Make a result array, and make it big enough.  Interleaving
        # writes to the output array randomly (well, not quite), so the
        # entire output array must be in memory.
        fmt = 'BH'[self.bitdepth > 8]
        a = array(fmt, [0]*vpr*self.height)
        source_offset = 0

        for xstart, ystart, xstep, ystep in _adam7:
            if xstart >= self.width:
                continue
            # The previous (reconstructed) scanline.  None at the
            # beginning of a pass to indicate that there is no previous
            # line.
            recon = None
            # Pixels per row (reduced pass image)
            ppr = int(math.ceil((self.width-xstart)/float(xstep)))
            # Row size in bytes for this pass.
            row_size = int(math.ceil(self.psize * ppr))
            for y in range(ystart, self.height, ystep):
                filter_type = raw[source_offset]
                source_offset += 1
                scanline = raw[source_offset:source_offset+row_size]
                source_offset += row_size
                recon = self.undo_filter(filter_type, scanline, recon)
                # Convert so that there is one element per pixel value
                flat = self.serialtoflat(recon, ppr)
                if xstep == 1:
                    assert xstart == 0
                    offset = y * vpr
                    a[offset:offset+vpr] = flat
                else:
                    offset = y * vpr + xstart * self.planes
                    end_offset = (y+1) * vpr
                    skip = self.planes * xstep
                    for i in range(self.planes):
                        a[offset+i:end_offset:skip] = \
                            flat[i::self.planes]
        return a