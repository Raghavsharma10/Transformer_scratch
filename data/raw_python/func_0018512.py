def cds(self):
        """just the parts of the exons that are translated"""
        ces = self.coding_exons
        if len(ces) < 1: return ces
        ces[0] = (self.cdsStart, ces[0][1])
        ces[-1] = (ces[-1][0], self.cdsEnd)
        assert all((s < e for s, e in ces))
        return ces