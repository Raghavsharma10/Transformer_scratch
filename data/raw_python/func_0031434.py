def readAlignments(self, reads):
        """
        Read lines of JSON from self._filename, convert them to read alignments
        and yield them.

        @param reads: An iterable of L{Read} instances, corresponding to the
            reads that were given to BLAST.
        @raise ValueError: If any of the lines in the file cannot be converted
            to JSON.
        @return: A generator that yields C{dark.alignments.ReadAlignments}
            instances.
        """
        if self._fp is None:
            self._open(self._filename)

        reads = iter(reads)

        try:
            for lineNumber, line in enumerate(self._fp, start=2):
                try:
                    record = loads(line[:-1])
                except ValueError as e:
                    raise ValueError(
                        'Could not convert line %d of %r to JSON (%s). '
                        'Line is %r.' %
                        (lineNumber, self._filename, e, line[:-1]))
                else:
                    try:
                        read = next(reads)
                    except StopIteration:
                        raise ValueError(
                            'Read generator failed to yield read number %d '
                            'during parsing of BLAST file %r.' %
                            (lineNumber - 1, self._filename))
                    else:
                        alignments = self._dictToAlignments(record, read)
                        yield ReadAlignments(read, alignments)
        finally:
            self._fp.close()
            self._fp = None