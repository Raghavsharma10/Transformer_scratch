def hsps(self):
        """
        Get all HSPs for all the alignments for all titles.

        @return: A generator yielding L{dark.hsp.HSP} instances.
        """
        return (hsp for titleAlignments in self.values()
                for alignment in titleAlignments for hsp in alignment.hsps)