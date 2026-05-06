def summary(self, sortOn=None):
        """
        Summarize all the alignments for this title.

        @param sortOn: A C{str} attribute to sort titles on. One of 'length',
            'maxScore', 'medianScore', 'readCount', or 'title'.
        @raise ValueError: If an unknown C{sortOn} value is given.
        @return: A generator that yields C{dict} instances as produced by
            C{TitleAlignments} (see class earlier in this file), sorted by
            C{sortOn}.
        """
        titles = self if sortOn is None else self.sortTitles(sortOn)

        for title in titles:
            yield self[title].summary()