def alignments(self):
        """
        Get alignments from the SAM/BAM file, subject to filtering.
        """
        referenceIds = self.referenceIds
        dropUnmapped = self.dropUnmapped
        dropSecondary = self.dropSecondary
        dropSupplementary = self.dropSupplementary
        dropDuplicates = self.dropDuplicates
        keepQCFailures = self.keepQCFailures
        storeQueryIds = self.storeQueryIds
        filterRead = self.filterRead
        minScore = self.minScore
        maxScore = self.maxScore
        scoreTag = self.scoreTag

        if storeQueryIds:
            self.queryIds = queryIds = set()

        lastAlignment = None
        count = 0
        with samfile(self.filename) as samAlignment:
            for count, alignment in enumerate(samAlignment.fetch(), start=1):
                if storeQueryIds:
                    queryIds.add(alignment.query_name)

                if minScore is not None or maxScore is not None:
                    try:
                        score = alignment.get_tag(scoreTag)
                    except KeyError:
                        continue
                    else:
                        if ((minScore is not None and score < minScore) or
                                (maxScore is not None and score > maxScore)):
                            continue

                # Secondary and supplementary alignments may have a '*'
                # (pysam returns this as None) SEQ field, indicating that
                # the previous sequence should be used. This is best
                # practice according to section 2.5.2 of
                # https://samtools.github.io/hts-specs/SAMv1.pdf So we use
                # the last alignment query and quality strings if we get
                # None as a query sequence.
                if alignment.query_sequence is None:
                    if lastAlignment is None:
                        raise InvalidSAM(
                            'pysam produced an alignment (number %d) with no '
                            'query sequence without previously giving an '
                            'alignment with a sequence.' % count)
                    # Use the previous query sequence and quality. I'm not
                    # making the call to _hardClip dependent on
                    # alignment.cigartuples (as in the else clause below)
                    # because I don't think it's possible for
                    # alignment.cigartuples to be None in this case. If we
                    # have a second match on a query, then it must be
                    # aligned to something (i.e., it cannot be unmapped
                    # with no CIGAR string). The assertion will tell us if
                    # this is ever not the case.
                    assert alignment.cigartuples
                    (alignment.query_sequence,
                     alignment.query_qualities, _) = _hardClip(
                         lastAlignment.query_sequence,
                         lastAlignment.query_qualities,
                         alignment.cigartuples)
                else:
                    lastAlignment = alignment
                    if alignment.cigartuples:
                        (alignment.query_sequence,
                         alignment.query_qualities, _) = _hardClip(
                             alignment.query_sequence,
                             alignment.query_qualities,
                             alignment.cigartuples)

                if ((filterRead is None or
                     filterRead(Read(alignment.query_name,
                                     alignment.query_sequence,
                                     alignment.qual))) and
                    not (
                        (referenceIds and
                         alignment.reference_name not in referenceIds) or
                        (alignment.is_unmapped and dropUnmapped) or
                        (alignment.is_secondary and dropSecondary) or
                        (alignment.is_supplementary and dropSupplementary) or
                        (alignment.is_duplicate and dropDuplicates) or
                        (alignment.is_qcfail and not keepQCFailures))):
                    yield alignment

        self.alignmentCount = count