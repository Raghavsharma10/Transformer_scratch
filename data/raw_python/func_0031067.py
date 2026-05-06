def writePathogenIndex(self, fp):
        """
        Write a file of pathogen indices and names, sorted by index.

        @param fp: A file-like object, opened for writing.
        """
        print('\n'.join(
            '%d %s' % (index, name) for (index, name) in
            sorted((index, name) for (name, index) in self._pathogens.items())
        ), file=fp)