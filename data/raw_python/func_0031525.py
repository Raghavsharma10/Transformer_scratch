def _dictToAlignments(self, diamondDict, read):
        """
        Take a dict (made by DiamondTabularFormatReader.records)
        and convert it to a list of alignments.

        @param diamondDict: A C{dict}, from records().
        @param read: A C{Read} instance, containing the read that DIAMOND used
            to create this record.
        @return: A C{list} of L{dark.alignment.Alignment} instances.
        """
        alignments = []
        getScore = itemgetter('bits' if self._hspClass is HSP else 'expect')

        for diamondAlignment in diamondDict['alignments']:
            alignment = Alignment(diamondAlignment['length'],
                                  diamondAlignment['title'])
            alignments.append(alignment)
            for diamondHsp in diamondAlignment['hsps']:
                score = getScore(diamondHsp)
                normalized = normalizeHSP(diamondHsp, len(read),
                                          self.diamondTask)
                hsp = self._hspClass(
                    score,
                    readStart=normalized['readStart'],
                    readEnd=normalized['readEnd'],
                    readStartInSubject=normalized['readStartInSubject'],
                    readEndInSubject=normalized['readEndInSubject'],
                    readFrame=diamondHsp['frame'],
                    subjectStart=normalized['subjectStart'],
                    subjectEnd=normalized['subjectEnd'],
                    readMatchedSequence=diamondHsp['query'],
                    subjectMatchedSequence=diamondHsp['sbjct'],
                    # Use blastHsp.get on identicalCount and positiveCount
                    # because they were added in version 2.0.3 and will not
                    # be present in any of our JSON output generated before
                    # that. Those values will be None for those JSON files,
                    # but that's much better than no longer being able to
                    # read all that data.
                    identicalCount=diamondHsp.get('identicalCount'),
                    positiveCount=diamondHsp.get('positiveCount'))

                alignment.addHsp(hsp)

        return alignments