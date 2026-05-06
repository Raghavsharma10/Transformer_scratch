def iter(self):
        """
        Extract DIAMOND records and yield C{ReadAlignments} instances.

        @return: A generator that yields C{ReadAlignments} instances.
        """
        # Note that self._reader is already initialized (in __init__) for
        # the first input file. This is less clean than it could be, but it
        # makes testing easier, since open() is then only called once for
        # each input file.

        reads = iter(self.reads)
        first = True

        for filename in self.filenames:
            if first:
                # The first file has already been opened, in __init__.
                first = False
                reader = self._reader
            else:
                reader = self._getReader(filename, self.scoreClass)

            for readAlignments in reader.readAlignments(reads):
                yield readAlignments

        # Any remaining query reads must have had no subject matches.
        for read in reads:
            yield ReadAlignments(read, [])