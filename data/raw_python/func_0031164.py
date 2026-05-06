def filter(self, readAlignments):
        """
        Filter a read's alignments.

        @param readAlignments: A C{ReadAlignments} instance.
        @return: A C{ReadAlignments} instance if the passed
            C{readAlignments} is not filtered out, else C{False}.
        """

        # Implementation notes:
        #
        # 1. The order in which we carry out the filtering actions can make
        #    a big difference in the result of this function. The current
        #    ordering is based on what seems reasonable - it may not be the
        #    best way to do things. E.g., if maxHspsPerHit is 1 and there
        #    is a title regex, which should we perform first?
        #
        #    We perform filtering based on alignment before that based on
        #    HSPs. That's because there's no point filtering all HSPs for
        #    an alignment that we end up throwing away anyhow.
        #
        # 2. This function could be made faster if it first looked at its
        #    arguments and dynamically created an acceptance function
        #    (taking a readAlignments as an argument). The acceptance
        #    function would run without examining the desired filtering
        #    settings on each call the way the current code does.
        #
        # 3. A better approach with readIdRegex would be to allow the
        #    passing of a regex object. Then the caller would make the
        #    regex with whatever flags they liked (e.g., case insensitive).

        #
        # Alignment-only (i.e., non-HSP based) filtering.
        #
        if self.limit is not None and self.count == self.limit:
            return False

        # Does the read have too many alignments?
        if (self.maxAlignmentsPerRead is not None and
                len(readAlignments) > self.maxAlignmentsPerRead):
            return False

        # Filter on the read id.
        if (self.readIdRegex and
                self.readIdRegex.search(readAlignments.read.id) is None):
            return False

        if self.titleFilter:
            # Remove alignments against sequences whose titles are
            # unacceptable.
            wantedAlignments = []
            for alignment in readAlignments:
                if (self.titleFilter.accept(alignment.subjectTitle) !=
                        TitleFilter.REJECT):
                    wantedAlignments.append(alignment)
            if wantedAlignments:
                readAlignments[:] = wantedAlignments
            else:
                return False

        # Only return alignments that are against sequences of the
        # desired length.
        minSequenceLen = self.minSequenceLen
        maxSequenceLen = self.maxSequenceLen
        if minSequenceLen is not None or maxSequenceLen is not None:
            wantedAlignments = []
            for alignment in readAlignments:
                length = alignment.subjectLength
                if not ((minSequenceLen is not None and
                         length < minSequenceLen) or
                        (maxSequenceLen is not None and
                         length > self.maxSequenceLen)):
                    wantedAlignments.append(alignment)
            if wantedAlignments:
                readAlignments[:] = wantedAlignments
            else:
                return False

        if self.taxonomy is not None:
            wantedAlignments = []
            for alignment in readAlignments:
                lineage = self.lineageFetcher.lineage(alignment.subjectTitle)
                if lineage:
                    for taxonomyIdAndScientificName in lineage:
                        if self.taxonomy in taxonomyIdAndScientificName:
                            wantedAlignments.append(alignment)
                else:
                    # No lineage info was found. Keep the alignment
                    # since we can't rule it out.  We could add another
                    # option to control this.
                    wantedAlignments.append(alignment)
            if wantedAlignments:
                readAlignments[:] = wantedAlignments
            else:
                return False

        if self.oneAlignmentPerRead and readAlignments:
            readAlignments[:] = [bestAlignment(readAlignments)]

        #
        # From here on we do only HSP-based filtering.
        #

        # Throw out any unwanted HSPs due to maxHspsPerHit.
        if self.maxHspsPerHit is not None:
            for alignment in readAlignments:
                hsps = alignment.hsps
                if len(hsps) > self.maxHspsPerHit:
                    alignment.hsps = hsps[:self.maxHspsPerHit]

        # Throw out HSPs whose scores are not good enough.
        if self.scoreCutoff is not None:
            wantedAlignments = []
            for alignment in readAlignments:
                hsps = alignment.hsps
                wantedHsps = []
                for hsp in hsps:
                    if hsp.betterThan(self.scoreCutoff):
                        wantedHsps.append(hsp)
                if wantedHsps:
                    alignment.hsps = wantedHsps
                    wantedAlignments.append(alignment)
            if wantedAlignments:
                readAlignments[:] = wantedAlignments
            else:
                return False

        # Throw out HSPs that don't match in the desired place on the
        # matched sequence.
        minStart = self.minStart
        maxStop = self.maxStop
        if minStart is not None or maxStop is not None:
            wantedAlignments = []
            for alignment in readAlignments:
                hsps = alignment.hsps
                wantedHsps = []
                for hsp in hsps:
                    if not ((minStart is not None and
                             hsp.readStartInSubject < minStart) or
                            (maxStop is not None and
                             hsp.readEndInSubject > maxStop)):
                        wantedHsps.append(hsp)
                if wantedHsps:
                    alignment.hsps = wantedHsps
                    wantedAlignments.append(alignment)
            if wantedAlignments:
                readAlignments[:] = wantedAlignments
            else:
                return False

        self.count += 1
        return readAlignments