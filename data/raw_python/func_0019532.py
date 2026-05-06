def compare_motifs(self, m1, m2, match="total", metric="wic", combine="mean", pval=False):
        """Compare two motifs.
        
        The similarity metric can be any of seqcor, pcc, ed, distance, wic, 
        chisq, akl or ssd. If match is 'total' the similarity score is 
        calculated for the whole match, including positions that are not 
        present in both motifs. If match is partial or subtotal, only the
        matching psotiions are used to calculate the score. The score of
        individual position is combined using either the mean or the sum.

        Note that the match and combine parameters have no effect on the seqcor
        similarity metric.      

        Parameters
        ----------
        m1 : Motif instance
            Motif instance 1.

        m2 : Motif instance
            Motif instance 2.

        match : str, optional
            Match can be "partial", "subtotal" or "total". Not all metrics use 
            this.

        metric : str, optional
            Distance metric.

        combine : str, optional
            Combine positional scores using "mean" or "sum". Not all metrics
            use this.

        pval : bool, optional
            Calculate p-vale of match.
        
        Returns
        -------
        score, position, strand 
        """
        if metric == "seqcor":
            return seqcor(m1, m2)
        elif match == "partial":
            if pval:
                return self.pvalue(m1, m2, "total", metric, combine, self.max_partial(m1.pwm, m2.pwm, metric, combine))
            elif metric in ["pcc", "ed", "distance", "wic", "chisq", "ssd"]:
                return self.max_partial(m1.pwm, m2.pwm, metric, combine)
            else:
                return self.max_partial(m1.pfm, m2.pfm, metric, combine)

        elif match == "total":
            if pval:
                return self.pvalue(m1, m2, match, metric, combine, self.max_total(m1.pwm, m2.pwm, metric, combine))
            elif metric in ["pcc", 'akl']:
                # Slightly randomize the weight matrix
                return self.max_total(m1.wiggle_pwm(), m2.wiggle_pwm(), metric, combine)
            elif metric in ["ed", "distance", "wic", "chisq", "pcc", "ssd"]:
                return self.max_total(m1.pwm, m2.pwm, metric, combine)
            else:
                return self.max_total(m1.pfm, m2.pfm, metric, combine)
                
        elif match == "subtotal":
            if metric in ["pcc", "ed", "distance", "wic", "chisq", "ssd"]:
                return self.max_subtotal(m1.pwm, m2.pwm, metric, combine)
            else:
                return self.max_subtotal(m1.pfm, m2.pfm, metric, combine)