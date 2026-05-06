def filter(self, minMatchingReads=None, minMedianScore=None,
               withScoreBetterThan=None, minNewReads=None, minCoverage=None,
               maxTitles=None, sortOn='maxScore'):
        """
        Filter the titles in self to create another TitlesAlignments.

        @param minMatchingReads: titles that are matched by fewer reads
            are unacceptable.
        @param minMedianScore: sequences that are matched with a median
            bit score that is less are unacceptable.
        @param withScoreBetterThan: if the best score for a title is not
            as good as this value, the title is not acceptable.
        @param minNewReads: The C{float} fraction of its reads by which a new
            title's read set must differ from the read sets of all previously
            seen titles in order for this title to be considered acceptably
            different (and therefore interesting).
        @param minCoverage: The C{float} minimum fraction of the title sequence
            that must be matched by at least one read.
        @param maxTitles: A non-negative C{int} maximum number of titles to
            keep. If more titles than this are present, titles will be sorted
            (according to C{sortOn}) and only the best will be retained.
        @param sortOn: A C{str} attribute to sort on, used only if C{maxTitles}
            is not C{None}. See the C{sortTitles} method below for the legal
            values.
        @raise: C{ValueError} if C{maxTitles} is less than zero or the value of
            C{sortOn} is unknown.
        @return: A new L{TitlesAlignments} instance containing only the
            matching titles.
        """
        # Use a ReadSetFilter only if we're checking that read sets are
        # sufficiently new.
        if minNewReads is None:
            readSetFilter = None
        else:
            if self.readSetFilter is None:
                self.readSetFilter = ReadSetFilter(minNewReads)
            readSetFilter = self.readSetFilter

        result = TitlesAlignments(
            self.readsAlignments, self.scoreClass, self.readSetFilter,
            importReadsAlignmentsTitles=False)

        if maxTitles is not None and len(self) > maxTitles:
            if maxTitles < 0:
                raise ValueError('maxTitles (%r) cannot be negative.' %
                                 maxTitles)
            else:
                # There are too many titles. Make a sorted list of them so
                # we loop through them (below) in the desired order and can
                # break when/if we've reached the maximum. We can't just
                # take the first maxTitles titles from the sorted list now,
                # as some of those titles might later be discarded by the
                # filter and then we'd return a result with fewer titles
                # than we should.
                titles = self.sortTitles(sortOn)
        else:
            titles = self.keys()

        for title in titles:
            # Test max titles up front, as it may be zero.
            if maxTitles is not None and len(result) == maxTitles:
                break

            titleAlignments = self[title]
            if (minMatchingReads is not None and
                    titleAlignments.readCount() < minMatchingReads):
                continue

            # To compare the median score with another score, we must
            # convert both values to instances of the score class used in
            # this data set so they can be compared without us needing to
            # know if numerically greater scores are considered better or
            # not.
            if (minMedianScore is not None and
                    self.scoreClass(titleAlignments.medianScore()) <
                    self.scoreClass(minMedianScore)):
                continue

            if (withScoreBetterThan is not None and not
                    titleAlignments.hasScoreBetterThan(withScoreBetterThan)):
                continue

            if (minCoverage is not None and
                    titleAlignments.coverage() < minCoverage):
                continue

            if (readSetFilter and not
                    readSetFilter.accept(title, titleAlignments)):
                continue

            result.addTitle(title, titleAlignments)

        return result