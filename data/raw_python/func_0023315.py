def write(self, outfile, rows):
        """Write a PNG image to the output file.  `rows` should be
        an iterable that yields each row in boxed row flat pixel
        format.  The rows should be the rows of the original image,
        so there should be ``self.height`` rows of ``self.width *
        self.planes`` values.  If `interlace` is specified (when
        creating the instance), then an interlaced PNG file will
        be written.  Supply the rows in the normal image order;
        the interlacing is carried out internally.

        .. note ::

          Interlacing will require the entire image to be in working
          memory.
        """

        if self.interlace:
            fmt = 'BH'[self.bitdepth > 8]
            a = array(fmt, itertools.chain(*rows))
            return self.write_array(outfile, a)

        nrows = self.write_passes(outfile, rows)
        if nrows != self.height:
            raise ValueError(
              "rows supplied (%d) does not match height (%d)" %
              (nrows, self.height))