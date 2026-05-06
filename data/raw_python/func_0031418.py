def sortTitles(self, by):
        """
        Sort titles by a given attribute and then by title.

        @param by: A C{str}, one of 'length', 'maxScore', 'medianScore',
            'readCount', or 'title'.
        @raise ValueError: If an unknown C{by} value is given.
        @return: A sorted C{list} of titles.
        """
        # First sort titles by the secondary key, which is always the title.
        titles = sorted(iter(self))

        # Then sort on the primary key (if any).
        if by == 'length':
            return sorted(
                titles, reverse=True,
                key=lambda title: self[title].subjectLength)
        if by == 'maxScore':
            return sorted(
                titles, reverse=True, key=lambda title: self[title].bestHsp())
        if by == 'medianScore':
            return sorted(
                titles, reverse=True,
                key=lambda title: self.scoreClass(self[title].medianScore()))
        if by == 'readCount':
            return sorted(
                titles, reverse=True,
                key=lambda title: self[title].readCount())
        if by == 'title':
            return titles

        raise ValueError('Sort attribute must be one of "length", "maxScore", '
                         '"medianScore", "readCount", "title".')