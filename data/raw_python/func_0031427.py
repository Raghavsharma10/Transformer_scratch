def accept(self, title, titleAlignments):
        """
        Return C{True} if the read id set in C{titleAlignments} is sufficiently
        different from all previously seen read sets.

        @param title: A C{str} sequence title.
        @param titleAlignments: An instance of L{TitleAlignment}.
        @return: A C{bool} indicating whether a title has an acceptably novel
            read set or not.
        """

        # Sanity check: titles can only be passed once.
        assert title not in self._titles, (
            'Title %r seen multiple times.' % title)

        readIds = titleAlignments.readIds()
        newReadsRequired = ceil(self._minNew * len(readIds))

        for readSet, invalidatedTitles in self._titles.values():
            if len(readIds - readSet) < newReadsRequired:
                # Add this title to the set of titles invalidated by this
                # previously seen read set.
                invalidatedTitles.append(title)
                return False

        # Remember the new read set and an empty list of invalidated titles.
        self._titles[title] = (readIds, [])

        return True