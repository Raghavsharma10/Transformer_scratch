def utr5(self):
        """
        return the 5' UTR if appropriate
        """
        if not self.is_coding or len(self.exons) < 2: return (None, None)
        if self.strand == "+":
            s, e = (self.txStart, self.cdsStart)
        else:
            s, e = (self.cdsEnd, self.txEnd)
        if s == e: return (None, None)
        return s, e