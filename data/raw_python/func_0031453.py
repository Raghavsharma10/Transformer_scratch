def summarizePosition(self, index):
        """
        Compute residue counts at a specific sequence index.

        @param index: an C{int} index into the sequence.
        @return: A C{dict} with the count of too-short (excluded) sequences,
            and a Counter instance giving the residue counts.
        """
        countAtPosition = Counter()
        excludedCount = 0

        for read in self:
            try:
                countAtPosition[read.sequence[index]] += 1
            except IndexError:
                excludedCount += 1

        return {
            'excludedCount': excludedCount,
            'countAtPosition': countAtPosition
        }