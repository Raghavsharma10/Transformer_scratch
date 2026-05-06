def queries(self, rcSuffix='', rcNeeded=False, padChar='-',
                queryInsertionChar='N', unknownQualityChar='!',
                allowDuplicateIds=False, addAlignment=False):
        """
        Produce padded (with gaps) queries according to the CIGAR string and
        reference sequence length for each matching query sequence.

        @param rcSuffix: A C{str} to add to the end of query names that are
            reverse complemented. This is added before the /1, /2, etc., that
            are added for duplicated ids (if there are duplicates and
            C{allowDuplicateIds} is C{False}.
        @param rcNeeded: If C{True}, queries that are flagged as matching when
            reverse complemented should have reverse complementing when
            preparing the output sequences. This must be used if the program
            that created the SAM/BAM input flags reversed matches but does not
            also store the reverse complemented query.
        @param padChar: A C{str} of length one to use to pad queries with to
            make them the same length as the reference sequence.
        @param queryInsertionChar:  A C{str} of length one to use to insert
            into queries when the CIGAR string indicates that the alignment
            of a query would cause a deletion in the reference. This character
            is inserted as a 'missing' query character (i.e., a base that can
            be assumed to have been lost due to an error) whose existence is
            necessary for the match to continue.
        @param unknownQualityChar: The character to put into the quality
            string when unknown bases are inserted in the query or the query
            is padded on the left/right with gaps.
        @param allowDuplicateIds: If C{True}, repeated query ids (due to
            secondary or supplemental matches) will not have /1, /2, etc.
            appended to their ids. So repeated ids may appear in the yielded
            FASTA.
        @param addAlignment: If C{True} the reads yielded by the returned
            generator will also have an C{alignment} attribute, being the
            C{pysam.AlignedSegment} for the query.
        @raises InvalidSAM: If a query has an empty SEQ field and either there
            is no previous alignment or the alignment is not marked as
            secondary or supplementary.
        @return: A generator that yields C{Read} instances that are padded
            with gap characters to align them to the length of the reference
            sequence. See C{addAlignment}, above, to yield reads with the
            corresponding C{pysam.AlignedSegment}.
        """
        referenceLength = self.referenceLength

        # Hold the count for each id so we can add /1, /2 etc to duplicate
        # ids (unless --allowDuplicateIds was given).
        idCount = Counter()

        MATCH_OPERATIONS = {CMATCH, CEQUAL, CDIFF}

        for lineNumber, alignment in enumerate(
                self.samFilter.alignments(), start=1):

            query = alignment.query_sequence
            quality = ''.join(chr(q + 33) for q in alignment.query_qualities)

            if alignment.is_reverse:
                if rcNeeded:
                    query = DNARead('id', query).reverseComplement().sequence
                    quality = quality[::-1]
                if rcSuffix:
                    alignment.query_name += rcSuffix

            # Adjust the query id if it's a duplicate and we're not allowing
            # duplicates.
            if allowDuplicateIds:
                queryId = alignment.query_name
            else:
                count = idCount[alignment.query_name]
                idCount[alignment.query_name] += 1
                queryId = alignment.query_name + (
                    '' if count == 0 else '/%d' % count)

            referenceStart = alignment.reference_start
            atStart = True
            queryIndex = 0
            referenceIndex = referenceStart
            alignedSequence = ''
            alignedQuality = ''

            for operation, length in alignment.cigartuples:

                # The operations are tested in the order they appear in
                # https://samtools.github.io/hts-specs/SAMv1.pdf It would be
                # more efficient to test them in order of frequency of
                # occurrence.
                if operation in MATCH_OPERATIONS:
                    atStart = False
                    alignedSequence += query[queryIndex:queryIndex + length]
                    alignedQuality += quality[queryIndex:queryIndex + length]
                elif operation == CINS:
                    # Insertion to the reference. This consumes query bases but
                    # we don't output them because the reference cannot be
                    # changed.  I.e., these bases in the query would need to be
                    # inserted into the reference.  Remove these bases from the
                    # query but record what would have been inserted into the
                    # reference.
                    atStart = False
                    self.referenceInsertions[queryId].append(
                        (referenceIndex,
                         query[queryIndex:queryIndex + length]))
                elif operation == CDEL:
                    # Delete from the reference. Some bases from the reference
                    # would need to be deleted to continue the match. So we put
                    # an insertion into the query to compensate.
                    atStart = False
                    alignedSequence += queryInsertionChar * length
                    alignedQuality += unknownQualityChar * length
                elif operation == CREF_SKIP:
                    # Skipped reference. Opens a gap in the query. For
                    # mRNA-to-genome alignment, an N operation represents an
                    # intron.  For other types of alignments, the
                    # interpretation of N is not defined. So this is unlikely
                    # to occur.
                    atStart = False
                    alignedSequence += queryInsertionChar * length
                    alignedQuality += unknownQualityChar * length
                elif operation == CSOFT_CLIP:
                    # Bases in the query that are not part of the match. We
                    # remove these from the query if they protrude before the
                    # start or after the end of the reference. According to the
                    # SAM docs, 'S' operations may only have 'H' operations
                    # between them and the ends of the CIGAR string.
                    if atStart:
                        # Don't set atStart=False, in case there's another 'S'
                        # operation.
                        unwantedLeft = length - referenceStart
                        if unwantedLeft > 0:
                            # The query protrudes left. Copy its right part.
                            alignedSequence += query[
                                queryIndex + unwantedLeft:queryIndex + length]
                            alignedQuality += quality[
                                queryIndex + unwantedLeft:queryIndex + length]
                            referenceStart = 0
                        else:
                            referenceStart -= length
                            alignedSequence += query[
                                queryIndex:queryIndex + length]
                            alignedQuality += quality[
                                queryIndex:queryIndex + length]
                    else:
                        unwantedRight = (
                            (referenceStart + len(alignedSequence) + length) -
                            referenceLength)

                        if unwantedRight > 0:
                            # The query protrudes right. Copy its left part.
                            alignedSequence += query[
                                queryIndex:queryIndex + length - unwantedRight]
                            alignedQuality += quality[
                                queryIndex:queryIndex + length - unwantedRight]
                        else:
                            alignedSequence += query[
                                queryIndex:queryIndex + length]
                            alignedQuality += quality[
                                queryIndex:queryIndex + length]
                elif operation == CHARD_CLIP:
                    # Some bases have been completely removed from the query.
                    # This (H) can only be present as the first and/or last
                    # operation. There is nothing to do as the bases are simply
                    # not present in the query string in the SAM/BAM file.
                    pass
                elif operation == CPAD:
                    # This is "silent deletion from the padded reference",
                    # which consumes neither query nor reference.
                    atStart = False
                else:
                    raise ValueError('Unknown CIGAR operation:', operation)

                if operation in _CONSUMES_QUERY:
                    queryIndex += length

                if operation in _CONSUMES_REFERENCE:
                    referenceIndex += length

            if queryIndex != len(query):
                # Oops, we did not consume the entire query.
                raise ValueError(
                    'Query %r not fully consumed when parsing CIGAR string. '
                    'Query %r (len %d), final query index %d, CIGAR: %r' %
                    (alignment.query_name, query, len(query), queryIndex,
                     alignment.cigartuples))

            # We cannot test we consumed the entire reference.  The CIGAR
            # string applies to (and exhausts) the query but is silent
            # about the part of the reference that lies to the right of the
            # aligned query.

            # Put gap characters before and after the aligned sequence so that
            # it is offset properly and matches the length of the reference.
            padRightLength = (referenceLength -
                              (referenceStart + len(alignedSequence)))
            paddedSequence = (padChar * referenceStart +
                              alignedSequence +
                              padChar * padRightLength)
            paddedQuality = (unknownQualityChar * referenceStart +
                             alignedQuality +
                             unknownQualityChar * padRightLength)

            read = Read(queryId, paddedSequence, paddedQuality)

            if addAlignment:
                read.alignment = alignment

            yield read