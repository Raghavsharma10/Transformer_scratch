def coverageInfo(self):
        """
        Return information about the bases found at each location in our title
        sequence.

        @return: A C{dict} whose keys are C{int} subject offsets and whose
            values are unsorted lists of (score, base) 2-tuples, giving all the
            bases from reads that matched the subject at subject location,
            along with the bit score of the matching read.
        """
        result = defaultdict(list)

        for titleAlignment in self:
            for hsp in titleAlignment.hsps:
                score = hsp.score.score
                for (subjectOffset, base, _) in titleAlignment.read.walkHSP(
                        hsp, includeWhiskers=False):
                    result[subjectOffset].append((score, base))

        return result