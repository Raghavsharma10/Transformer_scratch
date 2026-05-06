def _dictToAlignments(self, blastDict, read):
        """
        Take a dict (made by XMLRecordsReader._convertBlastRecordToDict)
        and convert it to a list of alignments.

        @param blastDict: A C{dict}, from convertBlastRecordToDict.
        @param read: A C{Read} instance, containing the read that BLAST used
            to create this record.
        @raise ValueError: If the query id in the BLAST dictionary does not
            match the id of the read.
        @return: A C{list} of L{dark.alignment.Alignment} instances.
        """
        if (blastDict['query'] != read.id and
                blastDict['query'].split()[0] != read.id):
            raise ValueError(
                'The reads you have provided do not match the BLAST output: '
                'BLAST record query id (%s) does not match the id of the '
                'supposedly corresponding read (%s).' %
                (blastDict['query'], read.id))

        alignments = []
        getScore = itemgetter('bits' if self._hspClass is HSP else 'expect')

        for blastAlignment in blastDict['alignments']:
            alignment = Alignment(blastAlignment['length'],
                                  blastAlignment['title'])
            alignments.append(alignment)
            for blastHsp in blastAlignment['hsps']:
                score = getScore(blastHsp)
                normalized = normalizeHSP(blastHsp, len(read),
                                          self.application)
                hsp = self._hspClass(
                    score,
                    readStart=normalized['readStart'],
                    readEnd=normalized['readEnd'],
                    readStartInSubject=normalized['readStartInSubject'],
                    readEndInSubject=normalized['readEndInSubject'],
                    readFrame=blastHsp['frame'][0],
                    subjectStart=normalized['subjectStart'],
                    subjectEnd=normalized['subjectEnd'],
                    subjectFrame=blastHsp['frame'][1],
                    readMatchedSequence=blastHsp['query'],
                    subjectMatchedSequence=blastHsp['sbjct'],
                    # Use blastHsp.get on identicalCount and positiveCount
                    # because they were added in version 2.0.3 and will not
                    # be present in any of our JSON output generated before
                    # that. Those values will be None for those JSON files,
                    # but that's much better than no longer being able to
                    # read all that data.
                    identicalCount=blastHsp.get('identicalCount'),
                    positiveCount=blastHsp.get('positiveCount'))

                alignment.addHsp(hsp)

        return alignments