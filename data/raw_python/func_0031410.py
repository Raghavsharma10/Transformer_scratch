def coverage(self):
        """
        Get the fraction of this title sequence that is matched by its reads.

        @return: The C{float} fraction of the title sequence matched by its
            reads.
        """
        intervals = ReadIntervals(self.subjectLength)
        for hsp in self.hsps():
            intervals.add(hsp.subjectStart, hsp.subjectEnd)
        return intervals.coverage()