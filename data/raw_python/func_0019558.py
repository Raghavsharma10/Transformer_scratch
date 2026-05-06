def to_meme(self):
        """Return motif formatted in MEME format
        
        Returns
        -------
        m : str
            String of motif in MEME format.
        """
        motif_id = self.id.replace(" ", "_")
        m = "MOTIF %s\n" % motif_id
        m += "BL   MOTIF %s width=0 seqs=0\n"% motif_id
        m += "letter-probability matrix: alength= 4 w= %s nsites= %s E= 0\n" % (len(self), np.sum(self.pfm[0]))
        m +="\n".join(["\t".join(["%s" % x for x in row]) for row in self.pwm])
        return m