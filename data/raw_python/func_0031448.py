def maximumORFLength(self, openORFs=True):
        """
        Return the length of the longest (possibly partial) ORF in a translated
        read. The ORF may originate or terminate outside the sequence, which is
        why the length is just a lower bound.
        """
        return max(len(orf) for orf in self.ORFs(openORFs))