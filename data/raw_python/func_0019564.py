def pwm_scan(self, fa, cutoff=0.9, nreport=50, scan_rc=True):
        """Scan sequences with this motif.

        Scan sequences from a FASTA object with this motif. Less efficient 
        than using a Scanner object. By setting the cutoff to 0.0 and 
        nreport to 1, the best match for every sequence will be returned.
        Only the position of the matches is returned.

        Parameters
        ----------
        fa : Fasta object
            Fasta object to scan.
        cutoff : float , optional
            Cutoff to use for motif scanning. This cutoff is not specifically
            optimized and the strictness will vary a lot with motif lengh.
        nreport : int , optional
            Maximum number of matches to report.
        scan_rc : bool , optional
            Scan the reverse complement. True by default.
        
        Returns
        -------
        matches : dict
            Dictionary with motif matches. Only the position of the matches is
            returned.
        """
        c = self.pwm_min_score() + (
            self.pwm_max_score() - self.pwm_min_score()) * cutoff        
        pwm = self.pwm
        matches = {}
        for name, seq in fa.items():
            matches[name] = [] 
            result = pfmscan(seq.upper(), pwm, c, nreport, scan_rc)
            for _,pos,_ in result:
                matches[name].append(pos)
        return matches