def writeSampleIndex(self, fp):
        """
        Write a file of sample indices and names, sorted by index.

        @param fp: A file-like object, opened for writing.
        """
        print('\n'.join(
            '%d %s' % (index, name) for (index, name) in
            sorted((index, name) for (name, index) in self._samples.items())
        ), file=fp)