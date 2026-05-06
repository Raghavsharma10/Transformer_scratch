def file_scanlines(self, infile):
        """
        Generates boxed rows in flat pixel format, from the input file
        `infile`.  It assumes that the input file is in a "Netpbm-like"
        binary format, and is positioned at the beginning of the first
        pixel.  The number of pixels to read is taken from the image
        dimensions (`width`, `height`, `planes`) and the number of bytes
        per value is implied by the image `bitdepth`.
        """

        # Values per row
        vpr = self.width * self.planes
        row_bytes = vpr
        if self.bitdepth > 8:
            assert self.bitdepth == 16
            row_bytes *= 2
            fmt = '>%dH' % vpr
            def line():
                return array('H', struct.unpack(fmt, infile.read(row_bytes)))
        else:
            def line():
                scanline = array('B', infile.read(row_bytes))
                return scanline
        for y in range(self.height):
            yield line()