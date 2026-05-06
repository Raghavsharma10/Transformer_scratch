def hsps(self):
        """
        Provide access to all HSPs for all alignments of all reads.

        @return: A generator that yields HSPs (or LSPs).
        """
        for readAlignments in self:
            for alignment in readAlignments:
                for hsp in alignment.hsps:
                    yield hsp