def iter(self):
        """
        Extract BLAST records and yield C{ReadAlignments} instances.

        For each file except the first, check that the BLAST parameters are
        compatible with those found (above, in __init__) in the first file.

        @return: A generator that yields C{ReadAlignments} instances.
        """
        # Note that self._reader is already initialized (in __init__) for
        # the first input file. This is less clean than it could be, but it
        # makes testing easier, since open() is then only called once for
        # each input file.

        count = 0
        reader = self._reader
        reads = iter(self.reads)
        first = True

        for blastFilename in self.blastFilenames:
            if first:
                # No need to check params in the first file. We already read
                # them in and stored them in __init__.
                first = False
            else:
                reader = self._getReader(blastFilename, self.scoreClass)
                differences = checkCompatibleParams(
                    self.params.applicationParams, reader.params)
                if differences:
                    raise ValueError(
                        'Incompatible BLAST parameters found. The parameters '
                        'in %s differ from those originally found in %s. %s' %
                        (blastFilename, self.blastFilenames[0], differences))

            for readAlignments in reader.readAlignments(reads):
                count += 1
                yield readAlignments

        # Make sure all reads were used.
        try:
            read = next(reads)
        except StopIteration:
            pass
        else:
            raise ValueError(
                'Reads iterator contained more reads than the number of BLAST '
                'records found (%d). First unknown read id is %r.' %
                (count, read.id))