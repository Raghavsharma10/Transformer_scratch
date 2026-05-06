def palette(self, alpha='natural'):
        """Returns a palette that is a sequence of 3-tuples or 4-tuples,
        synthesizing it from the ``PLTE`` and ``tRNS`` chunks.  These
        chunks should have already been processed (for example, by
        calling the :meth:`preamble` method).  All the tuples are the
        same size: 3-tuples if there is no ``tRNS`` chunk, 4-tuples when
        there is a ``tRNS`` chunk.  Assumes that the image is colour type
        3 and therefore a ``PLTE`` chunk is required.

        If the `alpha` argument is ``'force'`` then an alpha channel is
        always added, forcing the result to be a sequence of 4-tuples.
        """

        if not self.plte:
            raise FormatError(
                "Required PLTE chunk is missing in colour type 3 image.")
        plte = group(array('B', self.plte), 3)
        if self.trns or alpha == 'force':
            trns = array('B', self.trns or '')
            trns.extend([255]*(len(plte)-len(trns)))
            plte = map(operator.add, plte, group(trns, 1))
        return plte