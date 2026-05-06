def overlaps(self, other):
        """
        check for overlap with the other interval
        """
        if self.chrom != other.chrom: return False
        if self.start >= other.end: return False
        if other.start >= self.end: return False
        return True