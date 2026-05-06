def records(self):
        """
        Parse the DIAMOND output and yield records. This will be used to read
        original DIAMOND output (either from stdin or from a file) to turn the
        DIAMOND results into Python dictionaries that will then be stored in
        our JSON format.

        @return: A generator that produces C{dict}s containing 'alignments' and
            'query' C{str} keys.
        """
        with as_handle(self._filename) as fp:
            previousQtitle = None
            subjectsSeen = {}
            record = {}
            for line in fp:
                line = line[:-1]
                try:
                    (qtitle, stitle, bitscore, evalue, qframe, qseq,
                     qstart, qend, sseq, sstart, send, slen, btop, nident,
                     positive) = line.split('\t')
                except ValueError as e:
                    # We may not be able to find 'nident' and 'positives'
                    # because they were added in version 2.0.3 and will not
                    # be present in any of our JSON output generated before
                    # that. So those values will be None when reading
                    # DIAMOND output without those fields, but that's much
                    # better than no longer being able to read that data.
                    if six.PY2:
                        error = 'need more than 13 values to unpack'
                    else:
                        error = (
                            'not enough values to unpack (expected 15, '
                            'got 13)')
                    if str(e) == error:
                        (qtitle, stitle, bitscore, evalue, qframe,
                         qseq, qstart, qend, sseq, sstart, send, slen,
                         btop) = line.split('\t')
                        nident = positive = None
                    else:
                        raise
                hsp = {
                    'bits': float(bitscore),
                    'btop': btop,
                    'expect': float(evalue),
                    'frame': int(qframe),
                    'identicalCount': None if nident is None else int(nident),
                    'positiveCount': (
                        None if positive is None else int(positive)),
                    'query': qseq,
                    'query_start': int(qstart),
                    'query_end': int(qend),
                    'sbjct': sseq,
                    'sbjct_start': int(sstart),
                    'sbjct_end': int(send),
                }
                if previousQtitle == qtitle:
                    # We have already started accumulating alignments for this
                    # query.
                    if stitle not in subjectsSeen:
                        # We have not seen this subject before, so this is a
                        # new alignment.
                        subjectsSeen.add(stitle)
                        alignment = {
                            'hsps': [hsp],
                            'length': int(slen),
                            'title': stitle,
                        }
                        record['alignments'].append(alignment)
                    else:
                        # We have already seen this subject, so this is another
                        # HSP in an already existing alignment.
                        for alignment in record['alignments']:
                            if alignment['title'] == stitle:
                                alignment['hsps'].append(hsp)
                                break
                else:
                    # All alignments for the previous query id (if any)
                    # have been seen.
                    if previousQtitle is not None:
                        yield record

                    # Start building up the new record.
                    record = {}
                    subjectsSeen = {stitle}
                    alignment = {
                        'hsps': [hsp],
                        'length': int(slen),
                        'title': stitle,
                    }
                    record['alignments'] = [alignment]
                    record['query'] = qtitle

                    previousQtitle = qtitle

            # Yield the last record, if any.
            if record:
                yield record