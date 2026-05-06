def to_transfac(self):
        """Return motif formatted in TRANSFAC format
        
        Returns
        -------
        m : str
            String of motif in TRANSFAC format.
        """
        m = "%s\t%s\t%s\n" % ("DE", self.id, "unknown")
        for i, (row, cons) in enumerate(zip(self.pfm, self.to_consensus())):
            m += "%i\t%s\t%s\n" % (i, "\t".join([str(int(x)) for x in row]), cons)
        m += "XX"
        return m