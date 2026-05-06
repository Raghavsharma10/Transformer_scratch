def hasScoreBetterThan(self, score):
        """
        Is there an HSP with a score better than a given value?

        @return: A C{bool}, C{True} if there is at least one HSP in the
        alignments for this title with a score better than C{score}.
        """
        # Note: Do not assume that HSPs in an alignment are sorted in
        # decreasing order (as they are in BLAST output). If we could
        # assume that, we could just check the first HSP in each alignment.
        for hsp in self.hsps():
            if hsp.betterThan(score):
                return True
        return False