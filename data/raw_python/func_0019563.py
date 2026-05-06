def consensus_scan(self, fa):
        """Scan FASTA with the motif as a consensus sequence.

        Parameters
        ----------
        fa : Fasta object
            Fasta object to scan
        
        Returns
        -------
        matches : dict
            Dictionaru with matches.
        """
        regexp = "".join(["[" + "".join(self.iupac[x.upper()]) + "]" for x in self.to_consensusv2()])
        p = re.compile(regexp)
        matches = {}
        for name,seq in fa.items():
            matches[name] = [] 
            for match in p.finditer(seq):
                middle = (match.span()[1] + match.span()[0]) / 2
                matches[name].append(middle)
        return matches