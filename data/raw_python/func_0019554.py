def score_kmer(self, kmer):
        """Calculate the log-odds score for a specific k-mer.

        Parameters
        ----------
        kmer : str
            String representing a kmer. Should be the same length as the motif.
        
        Returns
        -------
        score : float
            Log-odd score.
        """
        if len(kmer) != len(self.pwm):
            raise Exception("incorrect k-mer length")
        
        score = 0.0
        d = {"A":0, "C":1, "G":2, "T":3}
        for nuc, row in zip(kmer.upper(), self.pwm):
            score += log(row[d[nuc]] / 0.25 + 0.01)

        return score