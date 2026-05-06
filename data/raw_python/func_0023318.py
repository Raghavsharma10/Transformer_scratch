def convert_pnm(self, infile, outfile):
        """
        Convert a PNM file containing raw pixel data into a PNG file
        with the parameters set in the writer object.  Works for
        (binary) PGM, PPM, and PAM formats.
        """

        if self.interlace:
            pixels = array('B')
            pixels.fromfile(infile,
                            (self.bitdepth/8) * self.color_planes *
                            self.width * self.height)
            self.write_passes(outfile, self.array_scanlines_interlace(pixels))
        else:
            self.write_passes(outfile, self.file_scanlines(infile))