def coverageCounts(self):
        """
        For each location in the subject, return a count of how many times that
        location is covered by a read.

        @return: a C{Counter} where the keys are the C{int} locations on the
            subject and the value is the number of times the location is
            covered by a read.
        """
        coverageCounts = Counter()
        for start, end in self._intervals:
            coverageCounts.update(range(max(0, start),
                                        min(self._targetLength, end)))
        return coverageCounts