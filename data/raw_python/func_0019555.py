def pfm_to_pwm(self, pfm, pseudo=0.001):
        """Convert PFM with counts to a PFM with fractions.

        Parameters
        ----------
        pfm : list
            2-dimensional list with counts.
        pseudo : float
            Pseudocount used in conversion.
        
        Returns
        -------
        pwm : list
            2-dimensional list with fractions.
        """
        return [[(x + pseudo)/(float(np.sum(row)) + pseudo * 4) for x in row] for row in pfm]